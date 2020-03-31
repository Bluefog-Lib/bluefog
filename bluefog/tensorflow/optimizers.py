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

import tensorflow as tf

from bluefog.tensorflow.mpi_ops import allreduce, broadcast, size
from bluefog.tensorflow.util import _executing_eagerly, _cache


@_cache
def _make_allreduce_grads_fn(name, device):
    def allreduce_grads(grads):
        with tf.name_scope(name + "_Allreduce"):
            ar_grads = [allreduce(grad, device=device)
                        if grad is not None else grad
                        for grad in grads]
        return ar_grads

    if _executing_eagerly():
        if hasattr(tf, 'function'):
            # TensorFlow 1.14.0+
            return tf.function(allreduce_grads)
        else:
            return tf.contrib.eager.defun(allreduce_grads)
    else:
        return allreduce_grads


@_cache
def _make_broadcast_group_fn():
    if _executing_eagerly():
        # Eager mode will parallelize independent control flow
        def broadcast_group(variables, root_rank):
            for var in sorted(variables, key=__name__):
                var.assign(broadcast(var, root_rank))

        if hasattr(tf, 'function'):
            # TensorFlow 1.14.0+
            return tf.function(broadcast_group)
        else:
            return tf.contrib.eager.defun(broadcast_group)
    else:
        # Graph mode requires an Op
        def broadcast_group(variables, root_rank):
            return tf.group(*[var.assign(broadcast(var, root_rank))
                              for var in variables])

        return broadcast_group


def broadcast_variables(variables, root_rank):
    """Broadcasts variables from root rank to all other processes.

    Arguments:
        variables: variables for broadcast
        root_rank: rank of the process from which global variables will be broadcasted
                   to all other processes.
    """
    broadcast_group = _make_broadcast_group_fn()
    return broadcast_group(variables, root_rank)


try:
    # TensorFlow 2.x
    _LegacyOptimizer = tf.compat.v1.train.Optimizer
except AttributeError:
    try:
        # TensorFlow 1.x
        _LegacyOptimizer = tf.train.Optimizer
    except AttributeError:
        # Future TensorFlow versions
        _LegacyOptimizer = None

if _LegacyOptimizer is not None:
    class _DistributedOptimizer(_LegacyOptimizer):
        """An optimizer that wraps another tf.Optimizer, using an allreduce to
        average gradient values before applying gradients to model weights."""

        def __init__(self, optimizer, name=None, use_locking=False, device=''):
            if name is None:
                name = "Distributed{}".format(type(optimizer).__name__)
            super(_DistributedOptimizer, self).__init__(
                name=name, use_locking=use_locking)

            self._optimizer = optimizer
            self._allreduce_grads = _make_allreduce_grads_fn(
                name, device)

        def compute_gradients(self, *args, **kwargs):
            """Compute gradients of all trainable variables.

            See Optimizer.compute_gradients() for more info.

            In DistributedOptimizer, compute_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = self._optimizer.compute_gradients(*args, **kwargs)
            if size() > 1:
                grads, vars_ = zip(*gradients)
                avg_grads = self._allreduce_grads(grads)
                return list(zip(avg_grads, vars_))
            else:
                return gradients

        def apply_gradients(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.apply_gradients(*args, **kwargs)

        def get_slot(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot(*args, **kwargs)

        def get_slot_names(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot_names(*args, **kwargs)

        def variables(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.variables(*args, **kwargs)


def DistributedOptimizer(optimizer, name=None, use_locking=False, device=''):
    """Construct a new DistributedOptimizer, which uses another optimizer
    under the hood for computing single-process gradient values and
    applying gradient updates after the gradient values have been averaged
    across all the Bluefog ranks.

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "Distributed" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.
      device:
        Device to be used for dense tensors. Uses GPU by default if CUDA is
        available
    """
    if isinstance(optimizer, _LegacyOptimizer):
        return _DistributedOptimizer(optimizer, name, use_locking, device)
    elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
        raise NotImplementedError
    else:
        raise ValueError('Provided optimizer doesn\'t inherit from either legacy '
                         'TensorFlow or Keras optimizer: %s' % optimizer)


if hasattr(tf, 'GradientTape'):
    class _DistributedGradientTape(tf.GradientTape):
        def __init__(self, tape, device,
                     persistent=False, watch_accessed_variables=True):
            if hasattr(tape, '_watch_accessed_variables'):
                super(self.__class__, self).__init__(
                    persistent, watch_accessed_variables)
            else:
                super(self.__class__, self).__init__(persistent)

            self._tape = tape
            self._allreduce_grads = _make_allreduce_grads_fn(
                'DistributedGradientTape', device)

        def gradient(self, target, sources, output_gradients=None):
            gradients = super(self.__class__, self).gradient(
                target, sources, output_gradients)
            if size() > 1:
                return self._allreduce_grads(gradients)
            else:
                return gradients

    def DistributedGradientTape(gradtape, device=''):
        """A tape that wraps another tf.GradientTape, using an allreduce to
        average gradient values before applying gradients to model weights.

        Args:
          gradtape:
            GradientTape to use for computing gradients and applying updates.
          device:
            Device to be used for dense tensors. Uses GPU by default if CUDA is
            available
        """
        cls = type(gradtape.__class__.__name__, (gradtape.__class__,),
                   dict(_DistributedGradientTape.__dict__))
        if hasattr(gradtape, '_watch_accessed_variables'):
            return cls(gradtape._tape, device, gradtape._persistent,
                       gradtape._watch_accessed_variables)
        else:
            return cls(gradtape._tape, device, gradtape._persistent)
