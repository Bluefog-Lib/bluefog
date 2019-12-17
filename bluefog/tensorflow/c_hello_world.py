import ctypes

import tensorflow as tf
from bluefog.common.util import get_ext_suffix
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

from bluefog.common.util import get_extension_full_path
import bluefog.common.topology_util as topology_util

from bluefog.tensorflow.mpi_ops import init, shutdown
from bluefog.tensorflow.mpi_ops import size, local_size, rank, local_rank
from bluefog.tensorflow.mpi_ops import load_topology, set_topology
from bluefog.tensorflow.mpi_ops import mpi_threads_supported
from bluefog.tensorflow.mpi_ops import allreduce, broadcast

full_path = get_extension_full_path(__file__, 'mpi_lib')

print("full_path: ", full_path)
MPI_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

MPI_LIB_CTYPES.bluefog_init()
rank = MPI_LIB_CTYPES.bluefog_rank()
local_rank = MPI_LIB_CTYPES.bluefog_local_rank()
size = MPI_LIB_CTYPES.bluefog_size()
local_size = MPI_LIB_CTYPES.bluefog_local_size()
print(
    f"Rank: {rank}, local rank: {local_rank} Size: {size}, local size: {local_size}")


def _load_library(name):
    """Loads a .so file containing the specified operators.

    Args:
      name: The name of the .so file to load.

    Raises:
      NotFoundError if were not able to load .so file.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    return library


MPI_LIB = _load_library('mpi_lib' + get_ext_suffix())


def random_uniform(*args, **kwargs):
    if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
        tf.random.set_seed(12345 + rank)
        return tf.random.uniform(*args, **kwargs)
    else:
        tf.set_random_seed(12345 + rank)
        return tf.random_uniform(*args, **kwargs)


dim = 1
dtype = tf.float32

tensor = random_uniform(
    [3] * dim, -100, 100, dtype=dtype)
print("Previous: ", tensor)
summed = allreduce(tensor, average=False)
print("Allreduce: ", summed)

result = broadcast(tf.ones(shape=[3]) * (rank+1), root_rank=0)
print("Broadcast: ", result)