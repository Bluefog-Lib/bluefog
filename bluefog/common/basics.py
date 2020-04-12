# Copyright 2020 Bluefog Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import List
import atexit
import contextlib
import ctypes
import logging
import networkx

import bluefog.common.util as util
import bluefog.common.topology_util as topology_util


logger = logging.getLogger('bluefog')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)-15s %(levelname)s  %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class BlueFogBasics(object):
    """Wrapper class for the basic BlueFog API."""

    def __init__(self, pkg_path, *args):
        full_path = util.get_extension_full_path(pkg_path, *args)
        self._topology = None
        self._MPI_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
        self._is_topo_weighted = False

    def init(self, topology: networkx.DiGraph = None,
             is_weighted: bool = False):
        """A function that initializes BlueFog.

        Args:
          topology: A networkx. DiGraph object to decide the topology. If not provided
            a default power_two_ring structure is used.
          is_weighted: If set to true, the neighbor ops like (win_sync, neighbor_allreduce) will
            execute the weighted average instead, where the weight is the value used in
            topology matrix (including self).
        """
        self._MPI_LIB_CTYPES.bluefog_init()
        self.set_topology(topology, is_weighted)
        atexit.register(self.shutdown)

    def shutdown(self) -> int:
        """A function that shuts BlueFog down."""
        self._MPI_LIB_CTYPES.bluefog_shutdown()
        self.topology = None

    def size(self) -> int:
        """A function that returns the number of BlueFog processes.

        Returns:
          An integer scalar containing the number of BlueFog processes.
        """
        size = self._MPI_LIB_CTYPES.bluefog_size()
        if size == -1:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return size

    def local_size(self) -> int:
        """A function that returns the number of BlueFog processes within the
        node the current process is running on.

        Returns:
          An integer scalar containing the number of local BlueFog processes.
        """
        local_size = self._MPI_LIB_CTYPES.bluefog_local_size()
        if local_size == -1:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return local_size

    def rank(self) -> int:
        """A function that returns the BlueFog rank of the calling process.

        Returns:
          An integer scalar with the BlueFog rank of the calling process.
        """
        rank = self._MPI_LIB_CTYPES.bluefog_rank()
        if rank == -1:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return rank

    def local_rank(self) -> int:
        """A function that returns the local BlueFog rank of the calling process, within the
        node that it is running on. For example, if there are seven processes running
        on a node, their local ranks will be zero through six, inclusive.

        Returns:
          An integer scalar with the local BlueFog rank of the calling process.
        """
        local_rank = self._MPI_LIB_CTYPES.bluefog_local_rank()
        if local_rank == -1:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return local_rank

    def unified_mpi_window_model_supported(self) -> bool:
        """Returns a boolean value to indicate the MPI_Win model is unified or not.
        Unfornuately, it is a collective call. We have to create a fake win to get
        this information.
        """
        is_unified = self._MPI_LIB_CTYPES.bluefog_unified_mpi_window_model_supported()
        if is_unified == -1:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return is_unified == 1

    def mpi_threads_supported(self) -> bool:
        """A function that returns a flag indicating whether MPI multi-threading is supported.

        If MPI multi-threading is supported, users may mix and match BlueFog usage with other
        MPI libraries, such as `mpi4py`.

        Returns:
          A boolean value indicating whether MPI multi-threading is supported.
        """
        mpi_threads_supported = self._MPI_LIB_CTYPES.bluefog_mpi_threads_supported()
        if mpi_threads_supported == -1:
            raise ValueError(
                "BlueFog has not been initialized; use bf.init().")
        return mpi_threads_supported

    def is_topo_weighted(self) -> bool:
        """A function that returns if the virtual topology weights are used

        Returns:
          A boolean value indicating if the topology weights are used.
        """
        return self._is_topo_weighted

    def load_topology(self) -> networkx.DiGraph:
        """A funnction that return the virtual topology MPI used.

        Returns:
            topology: networkx.DiGraph.
        """
        return self._topology

    def in_neighbor_ranks(self) -> List[int]:
        """Return the ranks of all in-neighbors.
        Notice: No matter self-loop is presented or not, self rank will not be included.

        Returns:
            List[int]: in_neighbor_ranks
        """
        if self._topology is None:
            return []
        _rank = self.rank()
        in_neighbor_ranks = [r for r in self._topology.predecessors(self.rank())
                             if r != _rank]
        return in_neighbor_ranks

    def out_neighbor_ranks(self) -> List[int]:
        """Return the ranks of all out-neighbors.
        Notice: No matter self-loop is presented or not, self rank will not be included.

        Returns:
            List[int]: out_neighbor_ranks
        """
        if self._topology is None:
            return []
        _rank = self.rank()
        out_neighbor_ranks = [r for r in self._topology.successors(self.rank())
                              if r != _rank]
        return out_neighbor_ranks

    def set_topology(self, topology: networkx.DiGraph = None,
                     is_weighted: bool = False) -> bool:
        """A funnction that set the virtual topology MPI used.

        Args:
          Topo: A networkx. DiGraph object to decide the topology. If not provided
            a default power_two_ring structure is used.
          is_weighted: If set to true, the neighbor ops win_sync and neighbor_allreduce will
            execute the weighted average instead, where the weights are the value used in
            topology matrix (including self weights). Note win_get/win_put/win_accumulate do
            not use this weight due to ambiguity.

        Returns:
            bool: Whether topology is set correctly or not.

        Example:
            >>> import bluefog.torch as bf
            >>> from bluefog.common import topology_util
            >>> bf.init()
            >>> bf.set_topology(topology_util.BiRingGraph(bf.size()))
        """
        if topology is None:
            topology = topology_util.PowerTwoRingGraph(size=self.size())
            if self.local_rank() == 0:
                logger.info(
                    "Topology is not specified. Default Power Two topology is used.")

        if not isinstance(topology, networkx.DiGraph):
            raise TypeError("topology must be a networkx.DiGraph obejct.")
        if topology_util.IsTopologyEquivalent(topology, self._topology):
            if self.local_rank() == 0:
                logger.debug(
                    "Topology to set is the same as old one. Skip the setting.")
            return True

        # We remove the self-rank for any cases because MPI graph_comm do not include it.
        destinations = [r for r in topology.successors(self.rank())
                        if r != self.rank()]
        sources = [r for r in topology.predecessors(self.rank())
                   if r != self.rank()]
        indegree = len(sources)
        outdegree = len(destinations)
        sources_type = ctypes.c_int * indegree
        destinations_type = ctypes.c_int * outdegree

        if not is_weighted:
            self._MPI_LIB_CTYPES.bluefog_set_topology.argtypes = (
                [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                 ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            )
            ret = self._MPI_LIB_CTYPES.bluefog_set_topology(
                indegree, sources_type(*sources),
                outdegree, destinations_type(*destinations))
        else:
            # Here the source_weights is a vector containing weights from source, i.e.,
            # (in-)neighbors, converted from the neighbor_weights dictionary.
            self_weight, neighbor_weights = topology_util.GetWeights(topology, self.rank())
            source_weights = [neighbor_weights[r] for r in sorted(neighbor_weights.keys())]
            source_weights_type = ctypes.c_float * indegree
            self._MPI_LIB_CTYPES.bluefog_set_topology_with_weights.argtypes = (
                [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                 ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                 ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
            )
            ret = self._MPI_LIB_CTYPES.bluefog_set_topology_with_weights(
                indegree, sources_type(*sources),
                outdegree, destinations_type(*destinations),
                self_weight, source_weights_type(*source_weights)
            )
        if ret != 1:
            if self.local_rank() == 0:
                logger.error(
                    "Cannot set topology correctly. Three common reasons caused this. \n"
                    "1. Has Bluefog been initialized? use bf.init(). \n"
                    "2. The win_create has been called. It is not allowed to change\n"
                    "   the topology after that. You can call win_free() to unregister\n"
                    "   all window object first, then set the topology. \n"
                    "3. Make sure all previous MPI ops are done. It is not allowed to \n"
                    "   change the topology while there is undone MPI ops."
                )
            return False
        self._topology = topology
        self._is_topo_weighted = is_weighted
        return True

    def timeline_start_activity(self, tensor_name: str, activity_name: str) -> bool:
        """A python interface to call the timeline for StartActivity.
        If you want to use this function, please make sure to turn on the timeline first by
        setting the ENV variable BLUEFOG_TIMELINE = {file_name}, or use
        bfrun --timeline-filename {file_name} ...

        Args:
            tensor_name (str): The activity associated tensor name.
            activity_name (str): The activity type.

        Returns:
            bool: Whether timeline is executed correctly or not.

        Example:
            >>> import bluefog.torch as bf
            >>> from bluefog.common.util import env
            >>> with env(BLUEFOG_TIMELINE="./timeline_file"):
            >>>     bf.init()
            >>> bf.timeline_start_activity(tensor_name, activity_name)
            >>> ...
            >>> bf.timeline_end_activity(tensor_name)
        """
        self._MPI_LIB_CTYPES.bluefog_timeline.argtypes = (
            [ctypes.c_bool, ctypes.c_char_p, ctypes.c_char_p]
        )
        ret = self._MPI_LIB_CTYPES.bluefog_timeline(
            True, tensor_name.encode("utf-8"), activity_name.encode('utf-8'))
        if ret != 1:
            logger.error("Cannot start activity in the timeline. "
                         "Most common reason is you didn't turn on the timeline function. "
                         "Use bfrun --timeline-filename file_name ... or "
                         "setting the ENV variable BLUEFOG_TIMELINE = file_name")
            return False
        return True

    def timeline_end_activity(self, tensor_name: str) -> bool:
        """A python interface to call the timeline for EndActivity.

        Please check comments in timeline_start_activity for more explanation.
        """
        self._MPI_LIB_CTYPES.bluefog_timeline.argtypes = (
            [ctypes.c_bool, ctypes.c_char_p, ctypes.c_char_p]
        )
        ret = self._MPI_LIB_CTYPES.bluefog_timeline(
            False, tensor_name.encode("utf-8"), "".encode('utf-8'))
        if ret != 1:
            logger.error("Cannot end activity in the timeline. Check "
                         "Most common reason is you didn't turn on the timeline function. "
                         "Use bfrun --timeline-filename file_name ... or "
                         "setting the ENV variable BLUEFOG_TIMELINE = file_name")
            return False
        return True

    @contextlib.contextmanager
    def timeline_context(self, tensor_name: str, activity_name: str):
        """Context manager for activating timeline record.
        If you want to use this function, please make sure to turn on the timeline first by
        setting the ENV variable BLUEFOG_TIMELINE = {file_name}, or use
        bfrun --timeline-filename {file_name} ...

        Args:
            tensor_name (str): The activity associated tensor name.
            activity_name (str): The activity type.

        Example:
            >>> with bf.timeline_context(tensor_name, activity_name):
            >>>     time.sleep(1.0)
        """
        self.timeline_start_activity(tensor_name, activity_name)
        try:
            yield
        finally:
            self.timeline_end_activity(tensor_name)
