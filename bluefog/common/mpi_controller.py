import queue
import sys
import traceback
from typing import Dict

import numpy

from bluefog.common.mpi_context import MPIContext
from bluefog.common.common import MPIOpsType

# Unlike raw C++ MPI, mpi4py has already initialize and gather the
# the ranks, sizes into python objects.


class MPIController:
    def __init__(self, tensor_queue: queue.Queue, window_queue: queue.Queue, 
                 mpi_context: MPIContext, ops_cnt: Dict):
        """Initialize the MPIController with given tensor queue and mpi context.

        Args:
            tensor_queue (queue.Queue): A thread-safe queue to accept the
              tensor information to communicate.
            window_queue (queue.Queue): A thread-safe queue to accept the
              window ops information to communicate.
            mpi_context (MPIContext): MPIContext. Defaults to None.
            ops_cnt (Dictionary [ops_name: int]): A diction counting how many MPI
                operations have been called.
        """
        self.mpi_context = mpi_context
        self.tensor_queue = tensor_queue
        self.window_queue = window_queue

        self.send_dst_ranks = list(
            self.mpi_context.topology.successors(self.rank))
        self.recv_src_ranks = list(
            self.mpi_context.topology.predecessors(self.rank))

        self._indegree, self._outdegree, self._is_weghted = (
            self.mpi_context.dist_graph_comm.Get_dist_neighbors_count())
        self.ops_cnt = ops_cnt

    @property
    def size(self):
        return self.mpi_context.comm.Get_size()

    @property
    def rank(self):
        return self.mpi_context.comm.Get_rank()

    @property
    def local_size(self):
        return self.mpi_context.local_comm.Get_rank()

    @property
    def local_rank(self):
        return self.mpi_context.local_comm.Get_rank()

    @property
    def graph_indgree(self):
        return self._indegree

    @property
    def graph_outdegree(self):
        return self._outdegree

    @property
    def graph_isweighted(self):
        return self._is_weghted

    @property
    def mpi_threads_supported(self):
        return self.mpi_context.mpi_threads_supported

    def _broadcast(self, buf_vector, root_rank, callback):
        self.mpi_context.comm.Bcast(buf_vector, root=root_rank)
        callback()

    def _allreduce(self, send_vector, recv_vector, callback):
        self.mpi_context.comm.Allreduce(
            send_vector, recv_vector, op=self.mpi_context.mpi_sum_op
        )
        callback()

    def _allgather(self, send_vector, recv_vector, callback):
        self.mpi_context.comm.Allgather(send_vector, recv_vector)
        callback()

    def _neighbor_allreduce(self, send_vector, recv_vector, callback):
        self.mpi_context.dist_graph_comm.Neighbor_allgather(
            send_vector, recv_vector)
        callback(recv_vector, self.graph_indgree)

    def _neighbor_allgather(self, send_vector, recv_vector, callback):
        self.mpi_context.dist_graph_comm.Neighbor_allgather(
            send_vector, recv_vector)
        callback()

    def _win_put(self, memory, dtype, dst, win):
        del dtype # How to use data type properly?
        win.Lock(dst, self.mpi_context.MPI.LOCK_SHARED)
        win.Put(memory, dst)
        win.Unlock(dst)

    def _win_get(self, memory, src, win):
        win.Lock(src, self.mpi_context.MPI.LOCK_SHARED)
        win.Get(memory, src)
        win.Unlock(src)

    def process_queue(self):
        # Use conti notify instead of loop?
        try:
            tensor_table_entry, comm_ops = self.tensor_queue.get(
                block=True, timeout=0.01)
            self.ops_cnt[comm_ops] += 1
            if comm_ops == MPIOpsType.ALLREDUCE:
                self._allreduce(
                    send_vector=tensor_table_entry.tensor,
                    recv_vector=tensor_table_entry.output,
                    #  tag=tensor_table_entry.tensor_name,
                    callback=tensor_table_entry.callback,
                )
            elif comm_ops == MPIOpsType.ALLGATHER:
                self._allgather(
                    send_vector=tensor_table_entry.tensor,
                    recv_vector=tensor_table_entry.output,
                    #  tag=tensor_table_entry.tensor_name,
                    callback=tensor_table_entry.callback,
                )
            elif comm_ops == MPIOpsType.BROADCASE:
                if self.rank == tensor_table_entry.root_rank:
                    buf_vector = tensor_table_entry.tensor
                else:
                    buf_vector = tensor_table_entry.output
                self._broadcast(
                    buf_vector=buf_vector,
                    root_rank=tensor_table_entry.root_rank,
                    # tag=tensor_table_entry.tensor_name,
                    callback=tensor_table_entry.callback,
                )
            elif comm_ops == MPIOpsType.NEIGHBOR_ALLREDUCE:
                # reallocate the space for output
                # TODO(ybc) Check this point later, here we assume tensor is numpy array.
                pad_width = tuple([(0, tensor_table_entry.tensor.shape[0]*(self.graph_indgree-1))] +
                                  [[0, 0]]*(len(tensor_table_entry.tensor.shape)-1))
                new_output = numpy.pad(tensor_table_entry.output, pad_width=pad_width,
                                       mode='constant', constant_values=-1)
                self._neighbor_allreduce(
                    send_vector=tensor_table_entry.tensor,
                    recv_vector=new_output,
                    #  tag=tensor_table_entry.tensor_name,
                    callback=tensor_table_entry.callback,
                )
            elif comm_ops == MPIOpsType.NEIGHBOR_ALLGATHER:
                self._neighbor_allgather(
                    send_vector=tensor_table_entry.tensor,
                    recv_vector=tensor_table_entry.output,
                    #  tag=tensor_table_entry.tensor_name,
                    callback=tensor_table_entry.callback,
                )
            else:
                raise ValueError(
                    "Unknown communication operations {}".format(comm_ops))
        except queue.Empty:
            # Just ignore the empty expection
            pass
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            raise e

    def process_window_queue(self):
        try:
            window_table_entry, win_ops = self.window_queue.get(
                block=True, timeout=0.01)
            self.ops_cnt[win_ops] += 1
            if win_ops == MPIOpsType.WIN_PUT:
                send_window_table_entry = (
                    self.mpi_context.GetWindowSendObject(window_table_entry.window_name))
                neighbor_rank_list = []
                if window_table_entry.dst_rank is None:
                    # Does not specify the dst_ranks, use all neighbors
                    neighbor_rank_list = send_window_table_entry.dst_rank
                elif isinstance(window_table_entry.dst_rank, list):
                    neighbor_rank_list = [
                        w for w in send_window_table_entry.dst_rank
                        if w.rank in window_table_entry.dst_rank
                    ]
                else:
                    raise ValueError(
                        "The rank in window_table_entry for WIN_GET ops " +
                        "should be either None or a list of int, but get {}".format(
                            window_table_entry.dst_rank)
                    )
                for neighbor_rank in neighbor_rank_list:
                    self._win_put(
                        memory=window_table_entry.memory,
                        dtype=window_table_entry.tensor_type,
                        dst=neighbor_rank,
                        win=send_window_table_entry.window_object
                    )
                window_table_entry.callback()
            elif win_ops == MPIOpsType.WIN_GET:
                send_window_table_entry = (
                    self.mpi_context.GetWindowSendObject(window_table_entry.window_name))
                neighbor_rank_list = []
                if window_table_entry.src_rank is None:
                    # Does not specify the dst_ranks, use all neighbors
                    neighbor_rank_list = send_window_table_entry.dst_rank
                elif isinstance(window_table_entry.src_rank, list):
                    neighbor_window_list = [
                        w for w in neighbor_window_list
                        if w.rank in window_table_entry.src_rank
                    ]
                else:
                    raise ValueError(
                        "The rank in window_table_entry for WIN_GET ops " +
                        "should be either None or a list of int, but get {}".format(
                            window_table_entry.src_rank)
                    )
                src_tensor_list = []
                for neighbor_rank in neighbor_rank_list:
                    for_neighbor_memory = numpy.zeros_like(window_table_entry.memory)
                    self._win_get(
                        memory=for_neighbor_memory,
                        src=neighbor_rank,
                        win=send_window_table_entry.window_object
                    )
                    src_tensor_list.append(for_neighbor_memory)
                window_table_entry.callback(src_tensor_list)
            else:
                raise ValueError(
                    "Unknown window operations {}".format(win_ops))
        except queue.Empty:
            # Just ignore the empty expection
            pass
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            raise e

    def start_receiver_for_neighbors(self):
        # 1. How to pre-define buf the with known size? Use IProb or pass in by context?
        # 2. Should we confine this function to receiving all parameters only or more general?
        # 3. Setup the callback function to trick when new vector is recived transform back to
        # GPU based item and compute the average?

        # for rank in self.recv_src_ranks:
        #     self.mpi_context.comm.Irecv(buf, source=rank)
        pass
