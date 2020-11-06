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

from contextlib import contextmanager
from typing import List, Dict, Optional

import torch

from bluefog.torch import mpi_lib  # C library
from bluefog.common.basics import BlueFogBasics, logger
from bluefog.common.topology_util import GetRecvWeights

_basics = BlueFogBasics(__file__, 'mpi_lib')

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank
load_topology = _basics.load_topology
is_topo_weighted = _basics.is_topo_weighted
set_topology = _basics.set_topology
in_neighbor_ranks = _basics.in_neighbor_ranks
out_neighbor_ranks = _basics.out_neighbor_ranks
mpi_threads_supported = _basics.mpi_threads_supported
unified_mpi_window_model_supported = _basics.unified_mpi_window_model_supported
is_homogeneous = _basics.is_homogeneous
nccl_built = _basics.nccl_built
set_skip_negotiate_stage = _basics.set_skip_negotiate_stage
get_skip_negotiate_stage = _basics.get_skip_negotiate_stage

timeline_context = _basics.timeline_context
timeline_start_activity = _basics.timeline_start_activity
timeline_end_activity = _basics.timeline_end_activity

# Schema: handle -> input, output
# We keep input in order to make sure it does not get garbage collected
# before the operation is finished.
_handle_map = {}

# Schema: handle -> name
_win_handle_map = {}

# Schema: name -> tensor
# Added in WinCreate, removed in WinFree, and referred by sync.
_win_map = {}


def _check_rank(rank_: int):
    assert isinstance(rank_, int), "Rank has to be an integer."
    assert rank_ >= 0, "Ranks must be an integer between 0 and size-1."
    assert rank_ < size(), "Ranks must be an integer between 0 and size-1."

def _check_function(function_factory, tensor, *args):
    function = function_factory(tensor, *args)
    if not hasattr(mpi_lib, function):
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    if not tensor.is_contiguous():
        raise ValueError('Tensor is required to be contiguous.')
    return function


def _allreduce_function_factory(tensor):
    return 'bluefog_torch_allreduce_nonblocking_' + tensor.type().replace('.', '_')


def _allreduce_nonblocking(tensor, output, average, is_hierarchical_local, name):
    function = _check_function(_allreduce_function_factory, tensor)
    if average:
        assert isinstance(tensor, (torch.HalfTensor, torch.FloatTensor, torch.DoubleTensor,
                                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
                                   torch.cuda.HalfTensor)), \
            "If average is set in allreduce, only float or double tensor is allowed."

    handle = getattr(mpi_lib, function)(tensor, output, average, is_hierarchical_local,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def allreduce(tensor: torch.Tensor, average: bool = True,
              is_hierarchical_local=False, name: Optional[str] = None) -> torch.Tensor:
    """
    A function that performs averaging or summation of the input tensor over all the
    Bluefog processes. The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        is_hierarchical_local: If set, allreduce is executed within one machine instead of
                global allreduce.
        name: A name of the reduction operation.

    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    handle = allreduce_nonblocking(tensor, average, is_hierarchical_local, name)
    return synchronize(handle)


def allreduce_nonblocking(tensor: torch.Tensor, average: bool = True,
                          is_hierarchical_local=False, name: Optional[str] = None) -> int:
    """
    A function that performs nonblocking averaging or summation of the input tensor
    over all the Bluefog processes. The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        is_hierarchical_local: If set, allreduce is executed within one machine instead of
                global allreduce.
        name: A name of the reduction operation.

    Returns:
        A handle to the allreduce operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new(tensor.shape)
    return _allreduce_nonblocking(tensor, output, average, is_hierarchical_local, name)


def allreduce_(tensor: torch.Tensor, average: bool = True,
               is_hierarchical_local=False, name: Optional[str] = None) -> torch.Tensor:
    """
    A function that performs averaging or summation of the input tensor over all the
    Bluefog processes. The operation is performed in-place.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        is_hierarchical_local: If set, allreduce is executed within one machine instead of
                global allreduce.
        name: A name of the reduction operation.

    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    handle = allreduce_nonblocking_(tensor, average, is_hierarchical_local, name)
    return synchronize(handle)


def allreduce_nonblocking_(tensor: torch.Tensor, average: bool = True,
                           is_hierarchical_local=False, name: Optional[str] = None) -> int:
    """
    A function that performs nonblocking averaging or summation of the input tensor
    over all the Bluefog processes. The operation is performed in-place.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        is_hierarchical_local: If set, allreduce is executed within one machine instead of
                global allreduce.
        name: A name of the reduction operation.

    Returns:
        A handle to the allreduce operation that can be used with `poll()` or
        `synchronize()`.
    """
    return _allreduce_nonblocking(tensor, tensor, average, is_hierarchical_local, name)


def _broadcast_function_factory(tensor):
    return 'bluefog_torch_broadcast_nonblocking_' + tensor.type().replace('.', '_')


def _broadcast_nonblocking(tensor, output, root_rank, name):
    function = _check_function(_broadcast_function_factory, tensor)
    handle = getattr(mpi_lib, function)(tensor, output, root_rank,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def broadcast(tensor: torch.Tensor, root_rank: int, name: Optional[str] = None) -> torch.Tensor:
    """
    A function that broadcasts the input tensor on root rank to the same input tensor
    on all other Bluefog processes. The input tensor is not modified.

    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.

    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.

    Returns:
        A tensor of the same shape and type as `tensor`, with the value broadcasted
        from root rank.
    """
    handle = broadcast_nonblocking(tensor, root_rank, name)
    return synchronize(handle)


def broadcast_nonblocking(tensor: torch.Tensor, root_rank: int, name: Optional[str] = None) -> int:
    """
    A function that nonblockingly broadcasts the input tensor on root rank to the same
    input tensor on all other Bluefog processes. The input tensor is not modified.

    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.

    Returns:
        A handle to the broadcast operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new(tensor.shape)
    return _broadcast_nonblocking(tensor, output, root_rank, name)


def broadcast_(tensor, root_rank, name=None) -> torch.Tensor:
    """
    A function that broadcasts the input tensor on root rank to the same input tensor
    on all other Bluefog processes. The operation is performed in-place.

    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.

    Returns:
        A tensor of the same shape and type as `tensor`, with the value broadcasted
        from root rank.
    """
    handle = broadcast_nonblocking_(tensor, root_rank, name)
    return synchronize(handle)


def broadcast_nonblocking_(tensor, root_rank, name=None) -> int:
    """
    A function that nonblockingly broadcasts the input tensor on root rank to the same
    input tensor on all other Bluefog processes. The operation is performed in-place.

    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.

    Returns:
        A handle to the broadcast operation that can be used with `poll()` or
        `synchronize()`.
    """
    return _broadcast_nonblocking(tensor, tensor, root_rank, name)


def _allgather_function_factory(tensor):
    return 'bluefog_torch_allgather_nonblocking_' + tensor.type().replace('.', '_')


def _allgather_nonblocking(tensor, output, name):
    function = _check_function(_allgather_function_factory, tensor)
    handle = getattr(mpi_lib, function)(tensor, output,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def allgather(tensor: torch.Tensor, name: Optional[str] = None) -> torch.Tensor:
    """
    A function that concatenates the input tensor with the same input tensor on
    all other Bluefog processes. The input tensor is not modified.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.

    Returns:
        A tensor of the same type as `tensor`, concatenated on dimension zero
        across all processes. The shape is identical to the input shape, except for
        the first dimension, which may be greater and is the sum of all first
        dimensions of the tensors in different Bluefog processes.
    """
    handle = allgather_nonblocking(tensor, name)
    return synchronize(handle)


def allgather_nonblocking(tensor: torch.Tensor, name: Optional[str] = None) -> int:
    """
    A function that nonblockingly concatenates the input tensor with the same input
    tensor on all other Bluefog processes. The input tensor is not modified.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.

    Returns:
        A handle to the allgather operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new()  # real size will be allocated later.
    return _allgather_nonblocking(tensor, output, name)


def _neighbor_allgather_function_factory(tensor):
    return 'bluefog_torch_neighbor_allgather_nonblocking_' + tensor.type().replace('.', '_')


def _neighbor_allgather_nonblocking(tensor, output, name):
    function = _check_function(_neighbor_allgather_function_factory, tensor)
    handle = getattr(mpi_lib, function)(tensor, output,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def neighbor_allgather(tensor: torch.Tensor, name: Optional[str] = None) -> torch.Tensor:
    """
    A function that concatenates the input tensor with the same input tensor on
    on all neighbor Bluefog processes (Not include self). The input tensor is not modified.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.

    Returns:
        A tensor of the same type as `tensor`, concatenated on dimension zero
        across all processes. The shape is identical to the input shape, except for
        the first dimension, which may be greater and is the sum of all first
        dimensions of the tensors in neighbor Bluefog processes.
    """
    handle = neighbor_allgather_nonblocking(tensor, name)
    return synchronize(handle)


def neighbor_allgather_nonblocking(tensor: torch.Tensor, name: Optional[str] = None) -> int:
    """
    A function that nonblockingly concatenates the input tensor with the same input
    tensor on all neighbor Bluefog processes (Not include self).
    The input tensor is not modified.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.

    Returns:
        A handle to the allgather operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new()  # real size will be allocated later.
    return _neighbor_allgather_nonblocking(tensor, output, name)


def _neighbor_allreduce_function_factory(tensor):
    return 'bluefog_torch_neighbor_allreduce_nonblocking_' + tensor.type().replace('.', '_')


def _neighbor_allreduce_nonblocking(tensor, output, self_weight, neighbor_weights,
                                    send_neighbors, enable_topo_check, name):
    function = _check_function(_neighbor_allreduce_function_factory, tensor)
    if send_neighbors is None:
        send_neighbors = []
        dynamic_neighbors_enabled = False
    elif len(set(send_neighbors)) != len(send_neighbors):
        raise ValueError("Argument send_neighbors should only contain the unique ranks.")
    elif self_weight is None or neighbor_weights is None:
        raise ValueError("Arguments self_weight and neighbor_weights should be presented if "
                         "enabling dynamic topology.")
    elif not send_neighbors:
        raise ValueError("Argument send_neighbors cannot be empty list but we plan to support "
                         "it in future.")
    else:
        dynamic_neighbors_enabled = True
    if self_weight is None and neighbor_weights is None:
        # Implying this is static graph.
        if is_topo_weighted():
            topology = load_topology()
            self_weight, neighbor_weights = GetRecvWeights(topology, rank())
            avg_computation = True
        else:
            weight = 1.0/(len(in_neighbor_ranks())+1)
            self_weight = weight
            neighbor_weights = {r: weight for r in in_neighbor_ranks()}
            avg_computation = False
    elif self_weight is not None and neighbor_weights is not None:
        if not isinstance(neighbor_weights, dict):
            raise ValueError("Argument neighbor_weights has to be a dictionary map from the "
                             "(in-)neighbor rank to the weights.")
        if not isinstance(self_weight, float):
            raise ValueError(
                "Argument self_weight has to be a float for self rank.")
        if not dynamic_neighbors_enabled and \
           not set(neighbor_weights.keys()).issubset(set(in_neighbor_ranks())):
            raise ValueError("The key of weights should only contain the ranks that belong to "
                             " in-neighbors and self rank.")
        uniform_weights = 1.0/(len(neighbor_weights)+1)
        avg_computation = False
        if abs(self_weight - uniform_weights) > 1e-6:
            avg_computation = True
        for n_weights in neighbor_weights.values():
            if abs(n_weights - uniform_weights) > 1e-6:
                avg_computation = True
                break
    else:
        raise ValueError("Arguments self_weight and neighbor_weights have to be presented at "
                         "the same time")
    is_hierarchical = False
    handle = getattr(mpi_lib, function)(tensor, output, self_weight, neighbor_weights,
                                        send_neighbors, dynamic_neighbors_enabled,
                                        enable_topo_check, avg_computation,
                                        is_hierarchical, name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def neighbor_allreduce(tensor: torch.Tensor,
                       self_weight: Optional[float] = None,
                       neighbor_weights: Optional[Dict[int, float]] = None,
                       send_neighbors: Optional[List[int]] = None,
                       enable_topo_check: bool = True,
                       name: Optional[str] = None) -> torch.Tensor:
    """
    A function that performs weighted averaging of the input tensor over the negihbors and itself
    in the Bluefog processes. The default behavior is (uniformly) average.

    The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to execute weighted average with neighbors.
        self_weight: The weight for self node, used with neighbor_weights.
        neighbor_weights: The weights for in-neighbor nodes, used with self weight.
            If neighbor_weights is presented, the return tensor will return the weighted average
            defined by these weights and the self_weight. If not, the return tensor will return
            the weighted average defined by the topology weights is provided or uniformly average.
            The data structure of weights should be {rank : weight} and rank has to belong to the
            (in-)neighbors.
        send_neighbors: The list of neighbor nodes to be sent to. If set to be None, assume the
            the current node sends to all of its (out-)neighbors. If having values, assume only
            part of (out-)neighbors will be sent to. In this mode, this node sends its value to
            partial neighbors listed in this variable in a dynamic graph, and `self_weight` and
            `neighbor_weights` must be present.
        enable_topo_check: When send_neighbors is present, enabling this option checks if the
            sending and recieving neighbors match with each other. Disabling this check can boost
            the performance.
        name: A name of the reduction operation.

    Returns:
        A tensor of the same shape and type as `tensor`,  across all processes.

    Note: self_weight and neighbor_weights must be presented at the same time.
    """
    if (self_weight is None and neighbor_weights is not None) or \
       (self_weight is not None and neighbor_weights is None):
        raise ValueError("Arguments self_weight and neighbor_weights have to be presented at "
                         "the same time")
    handle = neighbor_allreduce_nonblocking(tensor, self_weight, neighbor_weights,
                                            send_neighbors, enable_topo_check, name)
    return synchronize(handle)


def neighbor_allreduce_nonblocking(tensor: torch.Tensor,
                                   self_weight: Optional[float] = None,
                                   neighbor_weights: Optional[Dict[int, float]] = None,
                                   send_neighbors: Optional[List[int]] = None,
                                   enable_topo_check: bool = True,
                                   name: Optional[str] = None) -> int:
    """
    A function that nonblockingly performs weighted averaging of the input tensor over the
    negihbors and itself in the Bluefog processes. The default behavior is (uniformly) average.

    The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to execute weighted average with neighbors.
        self_weight: The weight for self node, used with neighbor_weights.
        neighbor_weights: The weights for in-neighbor nodes, used with self weight.
            If neighbor_weights is presented, the return tensor will return the weighted average
            defined by these weights and the self_weight. If not, the return tensor will return
            the weighted average defined by the topology weights is provided or uniformly average.
            The data structure of weights should be {rank : weight} and rank has to belong to the
            (in-)neighbors.
        send_neighbors: The list of neighbor nodes to be sent to. If set to be None, assume the
            the current node sends to all of its (out-)neighbors. If having values, assume only
            part of (out-)neighbors will be sent to. In this mode, this node sends its value to
            partial neighbors listed in this variable in a dynamic graph, and `self_weight` and
            `neighbor_weights` must be present.
        enable_topo_check: When send_neighbors is present, enabling this option checks if the
            sending and recieving neighbors match with each other. Disabling this check can boost
            the performance.
        name: A name of the neighbor_allreduce operation.

    Returns:
        A handle to the neighbor_allreduce operation that can be used with `poll()` or
        `synchronize()`.

    Note: self_weight and neighbor_weights must be presented at the same time.
    """
    if (self_weight is None and neighbor_weights is not None) or \
       (self_weight is not None and neighbor_weights is None):
        raise ValueError("Arguments self_weight and neighbor_weights have to be presented at "
                         "the same time")
    if send_neighbors is None:
        first_dim = tensor.shape[0] * len(in_neighbor_ranks())
    else:
        first_dim = tensor.shape[0] * len(neighbor_weights)
    new_shape = torch.Size([first_dim] + list(tensor.shape[1:]))
    output = tensor.new(new_shape)  # Pre-allocate the memory for the output.
    return _neighbor_allreduce_nonblocking(tensor, output, self_weight, neighbor_weights,
                                           send_neighbors, enable_topo_check, name=name)


def hierarchical_neighbor_allreduce(tensor: torch.Tensor,
                                    self_weight: float,
                                    neighbor_machine_weights: Dict[int, float],
                                    send_neighbor_machines: List[int],
                                    enable_topo_check: bool = False,
                                    name: Optional[str] = None) -> torch.Tensor:
    """
    A function that performs weighted averaging of the input tensor over the negihbor machines and
    itself in the Bluefog processes. It is similar to neighbor_allreduce. But each machine runs
    allreduce internal first to form a super node then executes the neighbor allreduce at machine
    level. The default behavior is (uniformly) average.

    The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Warning: This function should be called only under homogenerous environment, all machines have
    same number of Bluefog processes -- bf.local_size().

    Arguments:
        tensor: A tensor to execute weighted average with neighbor machines.
        self_weight: The weight for self node, used with neighbor_weights.
        neighbor_machine_weights: The weights for in-neighbor nodes, used with self weight.
            The data structure of weights should be {machine id : weight}.  All processes under
            same machine should specifiy the same weights dictionary.
        send_neighbor_machines: The list of neighbor machines to be sent to. All processes under
            same machine should specifiy the same machine id.
        enable_topo_check: When send_neighbors is present, enabling this option checks if the
            sending and recieving neighbors match with each other. Disabling this check can boost
            the performance.
        name: A name of the reduction operation.

    Returns:
        A tensor of the same shape and type as `tensor`,  across all processes.
    """
    # TODO(hhb) Implement the logics for topo check for hierarchical_neighbor_allreduce.

    # TODO(ybc) add check for self_weight and neighbor_machine_weights.
    if (self_weight is None and neighbor_machine_weights is not None) or \
       (self_weight is not None and send_neighbor_machines is None):
        raise ValueError("Arguments self_weight and neighbor_machine_weights have to "
                         "be presented at the same time")
    handle = hierarchical_neighbor_allreduce_nonblocking(
        tensor, self_weight, neighbor_machine_weights, send_neighbor_machines,
        enable_topo_check, name)
    return synchronize(handle)


def hierarchical_neighbor_allreduce_nonblocking(tensor: torch.Tensor,
                                                self_weight: float,
                                                neighbor_machine_weights: Dict[int, float],
                                                send_neighbor_machines: List[int],
                                                enable_topo_check: bool = False,
                                                name: Optional[str] = None) -> int:
    """
    A function that nonblockingly performs weighted averaging of the input tensor over the negihbor
    machines and itself in the Bluefog processes. It is similar to neighbor_allreduce. But
    each machine runs allreduce internal first to form a super node then executes
    the neighbor allreduce at machine level. The default behavior is (uniformly) average.

    The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Warning: This function should be called only under homogenerous environment, all machines have
    same number of Bluefog processes -- bf.local_size().

    Arguments:
        tensor: A tensor to execute weighted average with neighbor machines.
        self_weight: The weight for self node, used with neighbor_weights.
        neighbor_machine_weights: The weights for in-neighbor nodes, used with self weight.
            The data structure of weights should be {machine id : weight}.  All processes under
            same machine should specifiy the same weights dictionary.
        send_neighbor_machines: The list of neighbor machines to be sent to. All processes under
            same machine should specifiy the same machine id.
        enable_topo_check: When send_neighbors is present, enabling this option checks if the
            sending and recieving neighbors match with each other. Disabling this check can boost
            the performance.
        name: A name of the reduction operation.

    Returns:
        A handle to the hierarchical_neighbor_allreduce operation that can be used with `poll()` or
        `synchronize()`.
    """
    if (self_weight is None or neighbor_machine_weights is None):
        raise ValueError("Arguments self_weight and neighbor_weights cannot be empty or None.")
    if (self_weight is None and neighbor_machine_weights is not None) or \
       (self_weight is not None and neighbor_machine_weights is None):
        raise ValueError("Arguments self_weight and neighbor_weights have to be presented at "
                         "the same time")

    first_dim = tensor.shape[0] * len(neighbor_machine_weights)
    new_shape = torch.Size([first_dim] + list(tensor.shape[1:]))
    output = tensor.new(new_shape)  # Pre-allocate the memory for the output.

    return _hierarchical_neighbor_allreduce_nonblocking(
        tensor, output, self_weight, neighbor_machine_weights,
        send_neighbor_machines, enable_topo_check, name=name)


def _hierarchical_neighbor_allreduce_nonblocking(
        tensor, output, self_weight, neighbor_machine_weights,
        send_neighbor_machines, enable_topo_check, name):
    assert is_homogeneous, \
        "hierarchical_neighbor_allreduce should be used under homogeneous environment only"
    assert local_size() > 1, "If local size is 1, you should use neighbor allreduce directly."
    function = _check_function(_neighbor_allreduce_function_factory, tensor)
    if self_weight is not None and neighbor_machine_weights is not None:
        if not isinstance(neighbor_machine_weights, dict):
            raise ValueError("Argument neighbor_weights has to be a dictionary map from the "
                             "(in-)neighbor rank to the weights.")
        if not isinstance(self_weight, float):
            raise ValueError(
                "Argument self_weight has to be a float for self rank.")
        uniform_weights = 1.0/(len(neighbor_machine_weights)+1)
        avg_computation = False
        if abs(self_weight - uniform_weights) > 1e-6:
            avg_computation = True
        for n_weights in neighbor_machine_weights.values():
            if abs(n_weights - uniform_weights) > 1e-6:
                avg_computation = True
                break
    else:
        raise ValueError("Arguments self_weight and neighbor_weights cannot be empty or None.")

    if not send_neighbor_machines:
        raise ValueError("Argument send_neighbor_machines has to be presented and non-empty.")

    machine_size = size() // local_size()
    # Translate machine id into rank id.
    node_per_machine = local_size()
    neighbor_weights = {}
    for (m, weights) in neighbor_machine_weights.items():
        if m >= machine_size:
            raise ValueError(
                f"machine id is larger than number of machine we detected ({node_per_machine})."
                "Note it is 0-based index.")
        neighbor_weights[node_per_machine*m] = weights
    send_neighbors = [node_per_machine*m for m in send_neighbor_machines]
    tensor_buffer = tensor.detach().clone()
    is_hierarchical = True
    #TODO(ybc) support static machine topology case
    dynamic_neighbors_enabled = True
    handle = getattr(mpi_lib, function)(tensor_buffer, output, self_weight, neighbor_weights,
                                        send_neighbors, dynamic_neighbors_enabled, enable_topo_check,
                                        avg_computation, is_hierarchical,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor_buffer, output)
    return handle


def _pair_gossip_nonblocking_function_factory(tensor):
    return 'bluefog_torch_pair_gossip_nonblocking_' + tensor.type().replace('.', '_')


def _pair_gossip_nonblocking(tensor, output, target_rank, self_weight, pair_weight,
                             avg_computation, name):
    function = _check_function(
        _pair_gossip_nonblocking_function_factory, tensor)
    handle = getattr(mpi_lib, function)(tensor, output, target_rank,
                                        self_weight, pair_weight, avg_computation,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


# TODO(ybc) Consider active and passive version of pair gossip.
def pair_gossip(tensor: torch.Tensor, target_rank: int, self_weight: float = None,
                pair_weight: float = None, name: str = None) -> torch.Tensor:
    """
    A function that performs (weighted if specified) averaging of the input tensor and pair tensors
    in the Bluefog processes.

    Arguments:
        tensor: A tensor to pair_gossip.
        target_rank: The rank of pair node.
        self_weight: The weight for self node. If self_weight and pair_weight are not set,
            the returned tensor will be average value.
        pair_weight: The weight for pair node. If self_weight and pair_weight are not set,
            the returned tensor will be average value.
        name: A name of the pair_gossip operation.

    Returns:
        A tensor of the same shape and type as `tensor`.

    Note: 1. The input tensor is not modified.
          2. The pair process should call simultaneously with corresponding arguments.
          3. If not carefully used, it can lead to deadlock.
    """
    handle = pair_gossip_nonblocking(
        tensor, target_rank, self_weight, pair_weight, name)
    return synchronize(handle)


def pair_gossip_nonblocking(tensor: torch.Tensor, target_rank: int, self_weight: float = None,
                            pair_weight: float = None, name: str = None) -> int:
    """
    A function that nonblockingly performs (weighted if specified) averaging of the input tensor
     and pair tensors in the Bluefog processes.

    Arguments:
        tensor: A tensor to pair_gossip.
        target_rank: The rank of pair node.
        self_weight: The weight for self node. If self_weight and pair_weight are not set,
            the returned tensor will be average value.
        pair_weight: The weight for pair node. If self_weight and pair_weight are not set,
            the returned tensor will be average value.
        name: A name of the pair_gossip operation.

    Returns:
        A handle to the neighbor_allreduce operation that can be used with `poll()` or
        `synchronize()`.

    Note: 1. The input tensor is not modified.
          2. The pair process should call simultaneously with corresponding arguments.
          3. If not carefully used, it can lead to deadlock.
    """
    if pair_weight is None and self_weight is None:
        pair_weight = 0
        self_weight = 0
        avg_computation = True
    elif pair_weight is not None and self_weight is not None:
        avg_computation = False
    else:
        raise ValueError("self_weight and pair_weight have to be set at same time.")
    output = tensor.new(tensor.shape)
    return _pair_gossip_nonblocking(tensor, output, target_rank, self_weight,
                                    pair_weight, avg_computation, name)


def poll(handle: int) -> bool:
    """
    Polls an allreduce, neighbor_allreduce, etc operation handle to determine whether underlying
    nonblocking operation has completed. After `poll()` returns `True`, `wait()`
    will return without blocking.

    Arguments:
        handle: A handle returned by an allreduce, neighbor_allreduce, etc. nonblocking operation.

    Returns:
        A flag indicating whether the operation has completed.
    """
    return mpi_lib.bluefog_torch_poll(handle) != 0


def synchronize(handle: int) -> torch.Tensor:
    """
    Wait an allreduce, neighbor_allreduce, etc operation until
    it's completed. Returns the result of the operation.
    It is the same function as `wait()`.

    Args:
        handle: A handle returned by an allreduce, neighbor_allreduce, etc. nonblocking operation.

    Returns:
        torch.Tensor: An output tensor of the operation.
    """
    if handle not in _handle_map:
        raise ValueError("Cannot find handle to synchronize")
    mpi_lib.bluefog_torch_wait_and_clear(handle)
    _, output = _handle_map.pop(handle)
    return output


def wait(handle: int) -> torch.Tensor:
    """
    Wait an allreduce, neighbor_allreduce, etc operation until
    it's completed. Returns the result of the operation.
    It is just alias of `synchronize()` function.

    Args:
        handle: A handle returned by an allreduce, neighbor_allreduce, etc. nonblocking operation.

    Returns:
        torch.Tensor: An output tensor of the operation.
    """
    return synchronize(handle)


def barrier():
    """Barrier function to sychronize all MPI processes.

    After this function returns, it is guaranteed that all blocking functions
    before it is finished.
    """
    if get_skip_negotiate_stage():
        mpi_lib.bluefog_torch_barrier()
    else:
        allreduce(torch.Tensor([1.0]), name="barrier")

# MPI one sided ops, which will be useful in the asynchronized algorithm.


def _win_create_function_factory(tensor):
    return 'bluefog_torch_win_create_' + tensor.type().replace('.', '_')


def win_create(tensor: torch.Tensor, name: str, zero_init: bool = False) -> bool:
    """ Create MPI window for remote memoery access.

    The window is dedicated to the provided tensor only, which is identified by unqiue name.
    It is a blocking operation, which required all bluefog process involved.
    The initial values of MPI windows for neighbors are the same as input tensor unless
    zero_init is set to be true.

    Args:
        tensor (torch.Tensor): Provide the size, data type, and/or memory for window.
        name (str): The unique name to associate the window object.
        zero_init (boll): If set true, the buffer value initialize as zero instead of
            the value of tensor.

    Returns:
        bool: Indicate the creation succeed or not.

    Note: The window with same name across different bluefog processes should associate
    the tensor with same shape. Otherwise, the rest win_ops like win_update, win_put may
    encounter unrecoverable memory segmentation fault.
    """
    function = _check_function(_win_create_function_factory, tensor)
    if getattr(mpi_lib, function)(tensor, name, zero_init):
        _win_map[name] = tensor
        return True
    return False


def win_free(name: Optional[str] = None) -> bool:
    """ Free the MPI windows associated with name.

    Args:
        name (str): The unique name to associate the window object.
            If name is none, free all the window objects.

    Returns:
        bool: Indicate the free succeed or not.
    """
    if name is None:
        _win_map.clear()
        name = ''
    else:
        _win_map.pop(name)
    return getattr(mpi_lib, 'bluefog_torch_win_free')(name)


def _win_update_function_factory(tensor):
    return 'bluefog_torch_win_sync_' + tensor.type().replace('.', '_')


def win_update_then_collect(name: str, require_mutex: bool = True) -> torch.Tensor:
    """ A utility function to sync the neighbor buffers then accumulate all
    neighbor buffers' tensors into self tensor and clear the buffer.
    It is equivalent to

    >>> win_update(name, self_weight=1.0, neighbor_weights={neighbor: 1.0}, reset=True,
                   require_mutex=require_mutex)

    Args:
        name: The unique name to associate the window object.

    Returns:
        torch.Tensor: The average tensor of all neighbors' cooresponding tensors.
    """
    neighbor_weights = {r: 1.0 for r in in_neighbor_ranks()}
    return win_update(name, 1.0, neighbor_weights, reset=True, require_mutex=require_mutex)


def win_update(name: str,
               self_weight: Optional[float] = None,
               neighbor_weights: Optional[Dict[int, float]] = None,
               reset: bool = False, clone: bool = False,
               require_mutex: bool = False) -> torch.Tensor:
    """Locally synchronized the window objects and returned the reduced neighbor tensor.
    Note the returned tensor is the same tensor used in win_create and in-place modification
    is happened. During the update, a mutex for local variable is acquired.

    Args:
        name: The unique name to associate the window object.
        self_weight: the weight for self node, used with neighbor_weights.
        neighbor_weights: the weights for neighbor nodes, used with self_weight.
            If neighbor_weights is presented, the return tensor will return the weighted average
            defined by these weights and the self_weight. If not, the return tensor will return
            the weighted average defined by the topology weights if provided or mean value.
            The data structure of weights should be {rank : weight} and rank has to belong to
            the (in-)neighbors.
        reset: If reset is True, the buffer used to store the neighbor tensor included in
            neighbor_weights will be reset to zero.
            The reset is always happened after the weights computation.
            If neighbor_weights is not presented and reset is True, all the neighbor will be reset.
        clone: If set up to be true, the win_update result will return a new tensor instead of
            in-place change.
        require_mutex: If set to be true, the window mutex associated with local process will be
            acquired.

    Returns:
        torch.Tensor: The average tensor of all neighbors' cooresponding tensors.

    Note: Weights here will be useful if you need a dynamic weighted average, i.e. the weights
    change with the iterations. If static weight need, then setting the weights through the
    bf.set_topology(.., is_weighted=True) is a better choice.

    Note2: self_weight and neighbor_weights must be presented at the same time.
    """
    tensor = _win_map[name]
    if clone:
        tensor = tensor.clone()
    function = _check_function(_win_update_function_factory, tensor)

    if neighbor_weights is not None and self_weight is not None:
        # Pre-condition check for weights dictionary.
        if not isinstance(neighbor_weights, dict):
            raise ValueError("Argument neighbor_weights has to be a dictionary map from the "
                             "(in-)neighbor rank to the weights.")
        if not isinstance(self_weight, float):
            raise ValueError(
                "Argument self_weight has to be a float for self rank.")
        if not set(neighbor_weights.keys()).issubset(set(in_neighbor_ranks())):
            raise ValueError("The key of weights should only contain the ranks that belong to "
                             " in-neighbors and self rank.")
        avg_computation = True

    elif neighbor_weights is None and self_weight is None:
        if is_topo_weighted():
            topology = load_topology()
            self_weight, neighbor_weights = GetRecvWeights(topology, rank())
            avg_computation = True
        else:
            weight = 1.0/(len(in_neighbor_ranks())+1)
            self_weight = weight
            neighbor_weights = {r: weight for r in in_neighbor_ranks()}
            avg_computation = False
    else:
        raise ValueError("Arguments self_weight and neighbor_weights have to be presented at "
                         "the same time")

    if not getattr(mpi_lib, function)(tensor, name, self_weight, neighbor_weights,
                                      reset, avg_computation, require_mutex):
        raise RuntimeError("Cannot apply win_update on " + name)
    return tensor


def _win_put_function_factory(tensor):
    return 'bluefog_torch_win_put_' + tensor.type().replace('.', '_')


def win_put_nonblocking(tensor: torch.Tensor, name: str,
                        self_weight: Optional[float] = None,
                        dst_weights: Optional[Dict[int, float]] = None,
                        require_mutex: bool = False) -> int:
    """ Passively put the tensor into neighbor's shared window memory.
    This is a non-blocking function, which will return without waiting the
    win_put operation is really finished.

    Args:
        tesnor: The tensor that shares to neighbor.
        name: The unique name to associate the window object.
        self_weight: In-place multiply the weight to tensor (Happened after win_put send
            tensor information to neigbors), Default is 1.0.
        dst_weights: A dictionary that maps the destination ranks to the weight.
            Namely, {rank: weight} means put tensor * weight to the rank neighbor.
            If not provided, dst_weights will be set as all neighbor ranks defined by
            virtual topology with weight 1.
            Note dst_weights should only contain the ranks that belong to out-neighbors.
        require_mutex: If set to be true, out-neighbor process's window mutex will be
            acquired.

    Returns:
        A handle to the win_put operation that can be used with `win_poll()` or
        `win_wait()`.
    """
    function = _check_function(_win_put_function_factory, tensor)
    dst_weights = ({rank: 1.0 for rank in out_neighbor_ranks()}
                   if dst_weights is None else dst_weights)
    if self_weight is None:
        self_weight = 1.0
    if not set(dst_weights.keys()).issubset(set(out_neighbor_ranks())):
        raise ValueError(
            "The key of dst_weights should only contain ranks that "
            " belong to out-neighbors (self-rank is not allowed).")
    handle = getattr(mpi_lib, function)(
        tensor, name, self_weight, dst_weights, require_mutex)
    _win_handle_map[handle] = name
    return handle


def win_put(tensor: torch.Tensor, name: str,
            self_weight: Optional[float] = None,
            dst_weights: Optional[Dict[int, float]] = None,
            require_mutex: bool = False) -> bool:
    """ Passively put the tensor into neighbor's shared window memory.
    This is a blocking function, which will return until win_put operation
    is finished.

    Args:
        tensor: The tensor that shares to neighbor.
        name: The unique name to associate the window object.
        self_weight: In-place multiply the weight to tensor (Happened after win_put send
            tensor information to neigbors), Default is 1.0.
        dst_weights: A dictionary that maps the destination ranks to the weight.
            Namely, {rank: weight} means put tensor * weight to the rank neighbor.
            If not provided, dst_weights will be set as all neighbor ranks defined by
            virtual topology with weight 1.
            Note dst_weights should only contain the ranks that belong to out-neighbors.
        require_mutex: If set to be true, out-neighbor process's window mutex will be
            acquired.

    Returns:
        A bool value to indicate the put succeeded or not.
    """
    handle = win_put_nonblocking(tensor, name, self_weight, dst_weights, require_mutex)
    return win_wait(handle)


def win_get_nonblocking(name: str, src_weights: Optional[Dict[int, float]] = None,
                        require_mutex: bool = False) -> int:
    """ Passively get the tensor(s) from neighbors' shared window memory into
    local shared memory, which cannot be accessed in python directly.
    The win_update function is responsible for fetching that memeory.
    This is a non-blocking function, which will return without waiting the
    win_get operation is really finished.

    Args:
        name: The unique name to associate the window object.
        src_weights: A dictionary that maps the source ranks to the weight.
            Namely, {rank: weight} means get tensor from rank neighbor multipling the weight.
            If not provided, src_weights will be set as all neighbor ranks defined by
            virtual topology with weight 1.0.
            Note src_weights should only contain the in-neighbors only.
        require_mutex: If set to be true, out-neighbor process's window mutex will be
            acquired.

    Returns:
        A handle to the win_get operation that can be used with `win_poll()` or
        `win_wait()`.
    """
    function = "bluefog_torch_win_get"
    src_weights = ({rank: 1.0 for rank in in_neighbor_ranks()}
                   if src_weights is None else src_weights)
    if not set(src_weights.keys()).issubset(set(in_neighbor_ranks())):
        raise ValueError(
            "The key of src_weights should only containranks that "
            " belong to in-neighbors.")
    handle = getattr(mpi_lib, function)(name, src_weights, require_mutex)
    _win_handle_map[handle] = name
    return handle


def win_get(name: str, src_weights: Optional[Dict[int, float]] = None,
            require_mutex: bool = False) -> bool:
    """ Passively get the tensor(s) from neighbors' shared window memory into
    local shared memory, which cannot be accessed in python directly.
    The win_update function is responsible for fetching that memeory.
    This is a blocking function, which will return until win_get operation
    is finished.

    Args:
        tensor: A tensor to get the result, should have same shape and type of
            the window object associated with name.
        name: The unique name to associate the window object.
        src_weights: A dictionary that maps the source ranks to the weight.
            Namely, {rank: weight} means get tensor * weight to the rank neighbor.
            If not provided, src_weights will be set as all neighbor ranks defined by
            virtual topology with weight 1.0 / (neighbor_size+1).
            Note src_weights should only contain the ranks that either
            belong to int-neighbors or self.
        require_mutex: If set to be true, out-neighbor process's window mutex will be
            acquired.

    Returns:
        A bool value to indicate the get succeeded or not.
    """
    handle = win_get_nonblocking(name, src_weights, require_mutex)
    return win_wait(handle)


def _win_accumulate_function_factory(tensor):
    return 'bluefog_torch_win_accumulate_' + tensor.type().replace('.', '_')


def win_accumulate_nonblocking(tensor: torch.Tensor, name: str,
                               self_weight: Optional[float] = None,
                               dst_weights: Optional[Dict[int, float]] = None,
                               require_mutex: bool = False) -> bool:
    """ Passively accmulate the tensor into neighbor's shared window memory.
    Only SUM ops is supported now.
    This is a non-blocking function, which will return without waiting the
    win_accumulate operation is really finished.

    Args:
        tesnor: The tensor that shares to neighbor.
        name: The unique name to associate the window object.
        self_weight: In-place multiply the weight to tensor (Happened after win_accumulate
            send tensor information to neigbors), Default is 1.0.
        dst_weights: A dictionary that maps the destination ranks to the weight.
            Namely, {rank: weight} means accumulate tensor * weight to the rank neighbor.
            If not provided, dst_weights will be set as all neighbor ranks defined by
            virtual topology with weight 1.
            Note dst_weights should only contain the ranks that belong to out-neighbors.
        require_mutex: If set to be true, out-neighbor process's window mutex will be
            acquired.

    Returns:
        A handle to the win_accmulate operation that can be used with `win_poll()` or
        `win_wait()`.
    """
    function = _check_function(_win_accumulate_function_factory, tensor)
    dst_weights = ({rank: 1.0 for rank in out_neighbor_ranks()}
                   if dst_weights is None else dst_weights)
    if self_weight is None:
        self_weight = 1.0
    if not set(dst_weights.keys()).issubset(set(out_neighbor_ranks())):
        raise ValueError(
            "The key of dst_weights should only containranks that "
            " belong to out-neighbors (self-rank is not allowed).")
    handle = getattr(mpi_lib, function)(
        tensor, name, self_weight, dst_weights, require_mutex)
    _win_handle_map[handle] = name
    return handle


def win_accumulate(tensor: torch.Tensor, name: str,
                   self_weight: Optional[float] = None,
                   dst_weights: Optional[Dict[int, float]] = None,
                   require_mutex: bool = False) -> bool:
    """ Passively accmulate the tensor into neighbor's shared window memory.
    Only SUM ops is supported now.
    This is a blocking function, which will return until win_accumulate operation
    is finished.

    Args:
        tesnor: The tensor that shares to neighbor.
        name: The unique name to associate the window object.
        self_weight: In-place multiply the weight to tensor (Happened after win_accumulate
            send tensor information to neigbors), Default is 1.0.
        dst_weights: A dictionary that maps the destination ranks to the weight.
            Namely, {rank: weight} means accumulate tensor * weight to the rank neighbor.
            If not provided, dst_weights will be set as all neighbor ranks defined by
            virtual topology with weight 1.
            Note dst_weights should only contain the ranks that belong to out-neighbors.
        require_mutex: If set to be true, out-neighbor process's window mutex will be
            acquired.

    Returns:
        A bool value to indicate the accumulate succeeded or not.
    """
    handle = win_accumulate_nonblocking(
        tensor, name, self_weight, dst_weights, require_mutex)
    return win_wait(handle)


def win_poll(handle: int) -> bool:
    """Return whether the win ops identified by handle is done or not."""
    return mpi_lib.bluefog_torch_win_poll(handle) != 0


def win_wait(handle: int) -> bool:
    """Wait until the async win ops identified by handle is done."""
    if handle not in _win_handle_map:
        logger.warning("Win wait is called but the handle "
                       "is not found in the _win_handle_map.")
        return False
    mpi_lib.bluefog_torch_win_wait(handle)
    _ = _win_handle_map.pop(handle)
    return True


def get_win_version(name: str) -> Dict[int, int]:
    """ Get the version of tensor stored in the win buffer.

    Args:
        name: The unique name to get the associated window object.

    Returns:
        A dictionary maps from neighbor ranks to version. 0 means the latest
        tensor stored in win buffer has been read/sync. Non-negative value
        means the tensor has been updated through put or get before read/sync.
    """
    versions = [0] * size()
    returned_versions = mpi_lib.bluefog_torch_get_win_version(name, versions)
    neighbor_version = {r: returned_versions[r] for r in in_neighbor_ranks()}
    return neighbor_version


# Lock for MPI Open a passive RMA epcoh, which has nothing to do with mutex.
@contextmanager
def win_lock(name: str):
    """ win_lock context manager. Within the context, an RMA access epoch
    for its neihbor is created.
    Note The ops of win_get, win_accumulate, and win_put do not need win_lock context.

    Args:
        name: The name of existing MPI_win object. If not found, ValueError will raise.
    """
    _win_lock(name)
    try:
        yield
    finally:
        _win_unlock(name)


def _win_lock(name: str):
    if name not in _win_map:
        raise ValueError(
            "{} is not found in the registered window object.".format(name))
    mpi_lib.bluefog_torch_win_lock(name)


def _win_unlock(name: str):
    if name not in _win_map:
        raise ValueError(
            "{} is not found in the registered window object.".format(name))
    mpi_lib.bluefog_torch_win_unlock(name)


@contextmanager
def win_mutex(name: str, for_self=False, ranks: Optional[List[int]] = None):
    """ A win object implemented mutex context manager. Note, there are N distributed
    mutex over N corresponding processes.

    Args:
        name: Used to get the mutex for the window that registered by name.
        ranks: The mutex associated with the specified ranks is acquired.
            If not presented, the mutex with all out_neighbor ranks are acquired.
        for_self: If it is false, it will require the remote mutexes at processes ranks, which
            is specified by argument ranks). If it is true, it will require the self mutex.

    Example:
        >>> bf.win_create(tensor, name)
        >>> with win_mutex(name):
                tensor = bf.win_update_then_collect(name)
        >>> win_put(tensor, name)
    """
    _ranks = out_neighbor_ranks() if ranks is None else ranks
    _win_mutex_acquire(name, _ranks, for_self)
    try:
        yield
    finally:
        _win_mutex_release(name, _ranks, for_self)


def _win_mutex_acquire(name, ranks, for_self):
    mpi_lib.bluefog_torch_win_mutex_acquire(name, ranks, for_self)


def _win_mutex_release(name, ranks, for_self):
    mpi_lib.bluefog_torch_win_mutex_release(name, ranks, for_self)


def win_associated_p(name: str) -> float:
    """Return the associated correction P, used in Push-Sum algorithm, for each named window.

    Args:
        name (str): The unique name to associate the window object.

    Returns:
        float: The p value. (Initialized as 1.)
    """
    return mpi_lib.bluefog_torch_win_associated_p(name)


def turn_on_win_ops_with_associated_p():
    """Turn on the global state of win operations with associated p.

    If it is state is on, all win ops such as put, update, accumulate also apply on the
    associated p value as well.
    The default state is off.
    """
    mpi_lib.bluefog_torch_set_win_ops_with_associated_p_state(True)


def turn_off_win_ops_with_associated_p():
    """Turn off the global state of win operations with associated p."""
    mpi_lib.bluefog_torch_set_win_ops_with_associated_p_state(False)
