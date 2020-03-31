# Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

from bluefog.common.basics import BlueFogBasics
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
load_topology = _basics.load_topology
set_topology = _basics.set_topology
in_neighbor_ranks = _basics.in_neighbor_ranks
out_neighbor_ranks = _basics.out_neighbor_ranks

mpi_threads_supported = _basics.mpi_threads_supported
unified_mpi_window_model_supported = _basics.unified_mpi_window_model_supported

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


def allreduce(tensor: tf.Tensor, average: bool = True,
              name: str = None, device: str = '') -> tf.Tensor:
    """Perform an allreduce on a tf.Tensor or tf.IndexedSlices.

    This function performs a bandwidth-optimal ring allreduce on the input
    tensor. If the input is an tf.IndexedSlices, the function instead does an
    allgather on the values and the indices, effectively doing an allreduce on
    the represented tensor.

    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
                The shape of the input must be identical across all ranks.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.
        name: A name of the allreduce operation.
        device: Device to be used for dense tensors.

    Returns:
        A tensor of the same shape and type as `tensor`, summed across all
        processes.
    """
    if isinstance(tensor, tf.IndexedSlices):
        raise ValueError("Do not support Sparse or Indexed Slices Tensor yet.")
    else:
        with tf.device(device):
            bluefog_size = tf.cast(size(), dtype=tensor.dtype)
            summed_tensor = _allreduce(tensor, name)
            new_tensor = (summed_tensor /
                          bluefog_size) if average else summed_tensor
        return new_tensor


def broadcast(tensor: tf.Tensor, root_rank: int, name: str = None) -> tf.Tensor:
    """An op which broadcasts the input tensor on root rank to the same input tensor
    on all other Bluefog processes.

    The broadcast operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Bluefog processes for a given name. The broadcast
    will not start until all processes are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.

    Returns:
      A tensor of the same shape and type as `tensor`, with the value broadcasted
      from root rank.
    """
    if name is None and not _executing_eagerly():
        name = 'BluefogBroadcast_%s' % _normalize_name(tensor.name)
    return MPI_LIB.bluefog_broadcast(tensor, name=name, root_rank=root_rank)


@ops.RegisterGradient('BluefogBroadcast')
def _broadcast_grad(op, grad):
    """Gradient for broadcast op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    root_rank = op.get_attr('root_rank')
    grad_reduced = _allreduce(grad)
    if rank() != root_rank:
        return grad_reduced * 0
    return grad_reduced


def allgather(tensor: tf.Tensor, name: str = None) -> tf.Tensor:
    """An op which concatenates the input tensor with the same input tensor on
    all other Bluefog processes.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.

    Returns:
      A tensor of the same type as `tensor`, concatenated on dimension zero
      across all processes. The shape is identical to the input shape, except for
      the first dimension, which may be greater and is the sum of all first
      dimensions of the tensors in different Bluefog processes.
    """
    if name is None and not _executing_eagerly():
        name = 'BluefogAllgather_%s' % _normalize_name(tensor.name)
    return MPI_LIB.bluefog_allgather(tensor, name=name)


@ops.RegisterGradient('BluefogAllgather')
def _allgather_grad(op, grad):
    """Gradient for allgather op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    grad = _allreduce(grad)

    with tf.device('/cpu:0'):
        # Keep the tensor of split sizes on CPU.
        x = op.inputs[0]
        d0 = x.get_shape().as_list()[0]
        d = tf.convert_to_tensor([d0], dtype=tf.int32)

        s = size()
        d = tf.reshape(allgather(d), [s])

    splits = tf.split(grad, num_or_size_splits=d, axis=0)
    return splits[rank()]
