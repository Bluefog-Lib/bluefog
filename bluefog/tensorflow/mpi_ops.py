from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

from bluefog.common.basics_c import BlueFogBasics
from bluefog.common.util import get_ext_suffix
from bluefog.tensorflow.util import _executing_eagerly


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

_basics = BlueFogBasics(__file__, 'mpi_lib')

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank
mpi_threads_supported = _basics.mpi_threads_supported
load_topology = _basics.load_topology
set_topology = _basics.set_topology

# This function will create a default device map which includes all visible devices.
# Please run this function in a subprocess
def _check_has_gpu():
    import tensorflow
    return tensorflow.test.is_gpu_available()


def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)


def _allreduce(tensor, name=None):
    """An op which reduces an input tensor over all the Bluefog processes. The
    default reduction is a sum.

    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Bluefog processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.
    """
    if name is None and not _executing_eagerly():
        name = 'BluefogAllreduce_%s' % _normalize_name(tensor.name)
    return MPI_LIB.bluefog_allreduce(tensor, name=name)


@ops.RegisterGradient('BluefogAllreduce')
def _allreduce_grad(op, grad):
    """Gradient for allreduce op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    return _allreduce(grad)
