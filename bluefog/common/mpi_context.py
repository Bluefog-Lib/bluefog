import copy
import warnings
from typing import List, Tuple, Any

import numpy as np
import networkx as nx
from bluefog.common.common import WindowTableEntry
import bluefog.common.topology_util as topology_util

# TODO(ybc) Currently, using mpi4py as the interface to call the MPI function.
# In the future, we might want to use ctypes to call C++ MPI/NCCL/Gloo, etc directly and build
# an intermedia layer to unify the API so the python interface is always the same.


class MPIContextManager:
    """ A base class for managing MPI environment
    It can be derived if other frameworks to manage MPI environment.

    Note: Python has abs (abstract base classes) to enforce virtual function.
    """

    def EnvInitialize(self):
        raise NotImplementedError()

    def EnvFinalize(self):
        raise NotImplementedError()


class MPIContext:
    """ A temporary MPI context manager class.
    It should be independent of framework, tensor type, devices etc.

    Currently, we use mpi4py for fast development. So we assume
    all the exchaning information is numpy array.
    """

    __instance = None

    @staticmethod
    def getInstance(topology=None, MPI_context_manager=None):
        """ Static access method. """
        if MPIContext.__instance is None:
            MPIContext(topology, MPI_context_manager)
        return MPIContext.__instance

    def __init__(self, topology=None, MPI_context_manager=None):
        # we don't use MPI_context_manager yet. Everything is mpi4py for now.
        del MPI_context_manager

        if MPIContext.__instance is not None:
            warnings.warn(
                "MPIContext should be initialized once and is singleton!")
            return

        from mpi4py import MPI

        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self.topology = None
        self.send_dst_ranks, self.recv_src_ranks = None, None
        self._dist_graph_comm = None
        self.SetDistGraphComm(topology)

        self._local_comm = self._comm.Split_type(MPI.COMM_TYPE_SHARED)

        self.mpi_threads_supported = (
            self._MPI.Query_thread() == self._MPI.THREAD_MULTIPLE
        )

        self.initialized = True
        self.finalized = False
        # _win_dcit { name: [WindowTableEntry] }
        self._win_dict = {}
        self._inactive_win_dict = {}
        MPIContext.__instance = self

    def SetDistGraphComm(self, topology):
        # Create distributed graph communicators
        if isinstance(topology, (np.ndarray, np.matrix)):
            self.topology = nx.DiGraph(topology)
        elif isinstance(topology, nx.DiGraph):
            self.topology = topology
        elif topology is None:
            self.topology = topology_util.PowerTwoRingGraph(
                self._comm.Get_size())
        else:
            raise ValueError(
                "Provied topology should be either networkx.DiGraph or numpy.array")

        # Notice both send_dst_ranks and recv_src_ranks should not include `rank` iteself.
        self.send_dst_ranks = list(
            self.topology.successors(self._comm.Get_rank()))
        self.recv_src_ranks = list(
            self.topology.predecessors(self._comm.Get_rank()))

        self._dist_graph_comm = self._comm.Create_dist_graph_adjacent(
            sources=self.send_dst_ranks, destinations=self.recv_src_ranks
        )

    def WindowCreate(self, window_table_entry):
        """Register and create the mpi window object given shape and name.
        THe window object will be created multiple times. One for each in-coming neighbors.

        Args:
            window_table_entry (WindowTableEntry): containing the relevant information for
                mpi window objects.

        Raises:
            KeyError: When there are already registered mpi window for given name.
        """
        if window_table_entry.window_name in self._win_dict:
            raise KeyError("name {} has already been registered for mpi window.".format(
                window_table_entry.window_name))
        # in_neighbors, out_neighbors, is_weighted = self.dist_graph_comm.Get_dist_neighbors()
        # Although our topology is defined as directed graph, the window object is
        # always bi-directional, i.e. undirectional graph.

        mpi_windows = []
        inactive_mpi_windows = []
        # TODO(ybc) 1. Is it good to use WORLD_COMM to create window object?
        # What if use neighbor only comm?
        # 2. If one process failed but not window put dst, will it destory the window put?
        # 3. Do we need rank translation if neighbor only comm is used.

        # Recall win_create is a collective call executed by all processes in the group of comm.
        for n in range(self._comm.Get_size()):
            if n == self._comm.Get_rank(): # I am the sender.
                _window_table_entry = WindowTableEntry(
                    tensor_type=window_table_entry.tensor_type,
                    memory_shape=[],
                    window_name=window_table_entry.window_name,
                    dst_rank=self.send_dst_ranks,
                    src_rank=n
                )
                _window_table_entry.window_object = self._MPI.Win.Create(
                    memory=np.empty(0),
                    disp_unit=window_table_entry.disp_unit,
                    comm=self._comm)
                mpi_windows.append(_window_table_entry)
            elif n in self.recv_src_ranks: # I am the reciever.
                _window_table_entry = copy.deepcopy(window_table_entry)
                _window_table_entry.src_rank = n
                _window_table_entry.dst_rank = -1
                _window_table_entry.window_object = self._MPI.Win.Create(
                    memory=_window_table_entry.memory,
                    disp_unit=_window_table_entry.disp_unit,
                    comm=self._comm)
                mpi_windows.append(_window_table_entry)
            else:  # Just participate in a collective call.
                # Create a dummy win object.
                _window_table_entry = WindowTableEntry(
                    tensor_type=window_table_entry.tensor_type,
                    memory_shape=[],
                    window_name=window_table_entry.window_name,
                    dst_rank=-1,
                    src_rank=n,
                )
                _window_table_entry.window_object = self._MPI.Win.Create(
                    memory=np.empty(0),
                    disp_unit=window_table_entry.disp_unit,
                    comm=self._comm)
                inactive_mpi_windows.append(_window_table_entry)
        self._win_dict[window_table_entry.window_name] = mpi_windows
        self._inactive_win_dict[window_table_entry.window_name] = inactive_mpi_windows

    def WindowFree(self, name=None):
        if name is None:
            # Remove everything.
            for key_name in self._win_dict.keys():
                window_table_entries = self._win_dict.pop(key_name)
                window_table_entries.extend(self._inactive_win_dict.pop(key_name))
                for window_table_entry in sorted(window_table_entries,
                                                 key=lambda x: x.src_rank):
                    self._MPI.Win.Free(window_table_entry.window_object)
                    del window_table_entry
            self._win_dict.clear()
        elif (name in self._win_dict) and (name in self._inactive_win_dict):
            window_table_entries = self._win_dict.pop(name)
            window_table_entries.extend(self._inactive_win_dict.pop(name))
            for window_table_entry in sorted(window_table_entries,
                                             key=lambda x: x.src_rank):
                self._MPI.Win.Free(window_table_entry.window_object)
                del window_table_entry
        else:
            raise KeyError(
                "name {} has already been registered for mpi window.".format(name))

    def WindowSync(self, name) -> List[WindowTableEntry]:
        if name is None or name not in self._win_dict:
            raise KeyError(
                "name {} has not been registered for mpi window.".format(name))
        recv_window_table_entries = []
        for window_table_entry in self.GetWindowRecvObjectList(name):
            window_table_entry.window_object.Sync()
            recv_window_table_entries.append(window_table_entry)
        return recv_window_table_entries

    def GetWindowSendObject(self, name) -> WindowTableEntry:
        if name is None or name not in self._win_dict:
            raise KeyError(
                "name {} has not been registered for mpi window.".format(name))
        for window_table_entry in self._win_dict.get(name):
            if window_table_entry.src_rank == self._comm.Get_rank():
                return window_table_entry
        raise ValueError("Cannot find Sender MPI window Object.") # How to handle it?

    def GetWindowRecvObjectList(self, name) -> List[WindowTableEntry]:
        if name is None or name not in self._win_dict:
            raise KeyError(
                "name {} has not been registered for mpi window.".format(name))
        recv_window_table_entries = []
        for window_table_entry in self._win_dict.get(name):
            if window_table_entry.src_rank not in (-1, self._comm.Get_rank()):
                recv_window_table_entries.append(window_table_entry)
        return recv_window_table_entries

    def shutdown(self):
        self._MPI.Finalize()
        self.initialized = False
        self.finalized = True
        MPIContext.__instance = None

    @property
    def comm(self):
        return self._comm

    @property
    def local_comm(self):
        return self._local_comm

    @property
    def dist_graph_comm(self):
        return self._dist_graph_comm

    @property
    def mpi_sum_op(self):
        return self._MPI.SUM

    @property
    def MPI(self):
        return self._MPI
