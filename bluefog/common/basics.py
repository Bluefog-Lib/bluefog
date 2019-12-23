from typing import List
import atexit
import ctypes
import networkx

import bluefog.common.util as util
import bluefog.common.topology_util as topology_util


class BlueFogBasics(object):
    """Wrapper class for the basic BlueFog API."""

    def __init__(self, pkg_path, *args):
        full_path = util.get_extension_full_path(pkg_path, *args)
        self._topology = None
        self.MPI_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

    def init(self, topology: networkx.DiGraph = None, comm=None) -> None:
        """A function that initializes BlueFog.

        Args:
          Topo: A networkx. DiGraph object to decide the topology. If not provided
            a default power_two_ring structure is used.
          comm: List specifying ranks for the communicator, relative to the MPI_COMM_WORLD
            communicator OR the MPI communicator to use. Given communicator will be duplicated.
            If None, BlueFog will use MPI_COMM_WORLD Communicator.
        """
        del comm  # TODO(ybc) Allow to duplicate other communicator.
        self.MPI_LIB_CTYPES.bluefog_init()
        self.set_topology(topology)
        atexit.register(self.shutdown)

    def shutdown(self) -> int:
        """A function that shuts BlueFog down."""
        self.MPI_LIB_CTYPES.bluefog_shutdown()

    def size(self) -> int:
        """A function that returns the number of BlueFog processes.

        Returns:
          An integer scalar containing the number of BlueFog processes.
        """
        size = self.MPI_LIB_CTYPES.bluefog_size()
        if size == -1:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return size

    def local_size(self) -> int:
        """A function that returns the number of BlueFog processes within the
        node the current process is running on.

        Returns:
          An integer scalar containing the number of local BlueFog processes.
        """
        local_size = self.MPI_LIB_CTYPES.bluefog_local_size()
        if local_size == -1:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return local_size

    def rank(self) -> int:
        """A function that returns the BlueFog rank of the calling process.

        Returns:
          An integer scalar with the BlueFog rank of the calling process.
        """
        rank = self.MPI_LIB_CTYPES.bluefog_rank()
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
        local_rank = self.MPI_LIB_CTYPES.bluefog_local_rank()
        if local_rank == -1:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return local_rank

    def mpi_threads_supported(self) -> bool:
        """A function that returns a flag indicating whether MPI multi-threading is supported.

        If MPI multi-threading is supported, users may mix and match BlueFog usage with other
        MPI libraries, such as `mpi4py`.

        Returns:
          A boolean value indicating whether MPI multi-threading is supported.
        """
        mpi_threads_supported = self.MPI_LIB_CTYPES.bluefog_mpi_threads_supported()
        if mpi_threads_supported == -1:
            raise ValueError(
                "BlueFog has not been initialized; use bf.init().")
        return mpi_threads_supported

    def load_topology(self) -> networkx.DiGraph:
        """A funnction that return the virtual topology MPI used.

        Returns:
            topology: networkx.DiGraph.
        """
        return self._topology

    def in_neighbour_ranks(self) -> List[int]:
        """Return the ranks of all in-neighbors.

        Returns:
            List[int]: in_neighbour_ranks
        """
        if self._topology is None:
            return []
        in_neighbour_ranks = list(self._topology.predecessors(self.rank()))
        return in_neighbour_ranks

    def out_neighbor_ranks(self) -> List[int]:
        """Return the ranks of all out-neighbors.

        Returns:
            List[int]: out_neighbour_ranks
        """
        if self._topology is None:
            return []
        out_neighbor_ranks = list(self._topology.successors(self.rank()))
        return out_neighbor_ranks

    def set_topology(self, topology: networkx.DiGraph = None):
        """A funnction that set the virtual topology MPI used.

        Args:
          Topo: A networkx. DiGraph object to decide the topology. If not provided
            a default power_two_ring structure is used.

        Raises:
            RuntimeError: If topology can be set correctly.
        """
        if topology is None:
            print("Topology is not specified. Default Power Two topology is used.")
            topology = topology_util.PowerTwoRingGraph(size=self.size())

        destinations = list(topology.successors(self.rank()))
        sources = list(topology.predecessors(self.rank()))
        indegree = len(sources)
        outdegree = len(destinations)
        sources_type = ctypes.c_int * indegree
        destinations_type = ctypes.c_int * outdegree

        self.MPI_LIB_CTYPES.bluefog_set_topology.argtypes = (
            [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
             ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        )
        ret = self.MPI_LIB_CTYPES.bluefog_set_topology(
            indegree, sources_type(*sources),
            outdegree, destinations_type(*destinations))
        if ret != 1:
            raise RuntimeError(
                "Cannot set topology correctly. Has Bluefog been initialized? use bf.init()")
        self._topology = topology
