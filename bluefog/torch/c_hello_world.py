import ctypes
import time

# dummy import to avoid unknown link error.
import torch

from bluefog.common.util import get_extension_full_path
import bluefog.common.topology_util as topology_util

from bluefog.torch.mpi_ops_c import allreduce, allreduce_async
from bluefog.torch.mpi_ops_c import allgather, allgather_async
from bluefog.torch.mpi_ops_c import broadcast, broadcast_async
from bluefog.torch.mpi_ops_c import broadcast_, broadcast_async_
from bluefog.torch.mpi_ops_c import neighbor_allgather, neighbor_allgather_async
from bluefog.torch.mpi_ops_c import neighbor_allreduce, neighbor_allreduce_async
from bluefog.torch.mpi_ops_c import poll, synchronize
from bluefog.torch.mpi_ops_c import init, shutdown
from bluefog.torch.mpi_ops_c import size, local_size, rank, local_rank
from bluefog.torch.mpi_ops_c import load_topology, set_topology
from bluefog.torch.mpi_ops_c import mpi_threads_supported
from bluefog.torch.mpi_ops_c import win_create, win_free, win_sync
from bluefog.torch.mpi_ops_c import win_put, win_put_blocking
from bluefog.torch.mpi_ops_c import win_get, win_get_blocking
from bluefog.torch.mpi_ops_c import win_wait, win_poll

full_path = get_extension_full_path(__file__, 'mpi_lib')

print("full_path: ", full_path)
MPI_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

MPI_LIB_CTYPES.bluefog_init()
rank = MPI_LIB_CTYPES.bluefog_rank()
local_rank = MPI_LIB_CTYPES.bluefog_local_rank()
size = MPI_LIB_CTYPES.bluefog_size()
local_size = MPI_LIB_CTYPES.bluefog_local_size()
print(f"Rank: {rank}, local rank: {local_rank} Size: {size}, local size: {local_size}")


def SetTopology():
    global MPI_LIB_CTYPES
    topology = topology_util.PowerTwoRingGraph(size=size)

    destinations = list(topology.successors(rank))
    sources = list(topology.predecessors(rank))
    indegree = len(sources)
    outdegree = len(destinations)
    sources_type = ctypes.c_int * indegree
    destinations_type = ctypes.c_int * outdegree
    print(rank, ":  ", indegree, outdegree, sources, destinations)
    MPI_LIB_CTYPES.bluefog_set_topology.argtypes = (
        [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
         ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    )
    ret = MPI_LIB_CTYPES.bluefog_set_topology(
        indegree, sources_type(*sources),
        outdegree, destinations_type(*destinations))
    assert ret == 1
SetTopology()

result = allreduce(torch.Tensor([[1, 2, 3], [4, 5, 6]]).mul_(rank))
if rank == 0:
    print("Allreduce: ", result)

result = broadcast(torch.Tensor([1]).mul(rank+3), root_rank=0)
print("Broadcast: ", result)

result = allgather(torch.Tensor([[1, 2, 3], [4, 5, 6]]).mul(rank+1))
print("Allgather: ", result)

result = neighbor_allgather(torch.Tensor([[1, 2, 3], [4, 5, 6]]).mul(rank+1))
print(rank, "--neighbor_allgather: ", result)

result = neighbor_allreduce(torch.Tensor([[1, 2, 3], [4, 5, 6]]).mul(rank+1), average=True)
print(rank, "--neighbor_allreduce: ", result)

assert win_create(torch.Tensor([[1, 2, 3], [4, 5, 6]]).mul(rank+1), name="win_test")
result = win_sync(name="win_test")
print(rank, "--win_sync: ", result)
win_put_blocking(torch.Tensor([[1, 2, 3], [4, 5, 6]]).mul(rank+1), name="win_test")
time.sleep(0.01)
result = win_sync(name="win_test")
print(rank, "--win_sync: ", result)
win_get(torch.Tensor([[1, 2, 3], [4, 5, 6]]).mul(rank+1),
        name="win_test", average=True)

MPI_LIB_CTYPES.bluefog_shutdown()
