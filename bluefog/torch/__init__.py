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

import collections
import os
import torch
from bluefog.common.util import check_extension
from bluefog.torch.optimizers import (
    DistributedAllreduceOptimizer, DistributedConsensusOptimizer, DistributedBluefogOptimizer)

check_extension('bluefog.torch', __file__, 'mpi_lib')

from bluefog.torch.mpi_ops import init, shutdown
from bluefog.torch.mpi_ops import size, local_size, rank, local_rank
from bluefog.torch.mpi_ops import load_topology, set_topology
from bluefog.torch.mpi_ops import in_neighbor_ranks, out_neighbor_ranks
from bluefog.torch.mpi_ops import mpi_threads_supported
from bluefog.torch.mpi_ops import unified_mpi_window_model_supported

from bluefog.torch.mpi_ops import allreduce, allreduce_async
from bluefog.torch.mpi_ops import allgather, allgather_async
from bluefog.torch.mpi_ops import broadcast, broadcast_async
from bluefog.torch.mpi_ops import broadcast_, broadcast_async_
from bluefog.torch.mpi_ops import neighbor_allgather, neighbor_allgather_async
from bluefog.torch.mpi_ops import neighbor_allreduce, neighbor_allreduce_async
from bluefog.torch.mpi_ops import poll, synchronize, barrier

from bluefog.torch.mpi_ops import win_create, win_free
from bluefog.torch.mpi_ops import win_sync, win_sync_then_collect
from bluefog.torch.mpi_ops import win_put_async, win_put
from bluefog.torch.mpi_ops import win_get_async, win_get
from bluefog.torch.mpi_ops import win_accumulate_async, win_accumulate
from bluefog.torch.mpi_ops import win_wait, win_poll
from bluefog.torch.mpi_ops import win_lock, win_mutex

from bluefog.torch.mpi_ops import timeline_start_activity, timeline_end_activity
from bluefog.torch.mpi_ops import timeline_context

def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the ``model.state_dict()``,
    ``model.named_parameters()``, or ``model.parameters()``.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError("invalid params of type: %s" % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        handle = broadcast_async_(p, root_rank, name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        raise ValueError("cannot broadcast torch.optim.LBFGS state")

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if not state_dict["state"]:
        for group in optimizer.param_groups:
            for p in group["params"]:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedAllreduceOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if not state_dict["state"]:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict["state"][pid][name] = t(p.numpy()[0])

        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(
                option_tensor.numpy()[0], dtypes
            )

        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict["param_groups"]):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == "params":
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = "%s.%d" % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(
                index, option_key, option_tensor, dtypes
            )
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group["params"]:
            param_state = state_dict["state"][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = "%s.%d" % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()
