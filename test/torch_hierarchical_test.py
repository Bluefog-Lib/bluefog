import itertools
import os

import numpy as np
import pytest
import torch
import networkx as nx

import bluefog.torch as bf
from common import mpi_env_rank_and_size

EPSILON = 1e-6
TEST_ON_GPU = torch.cuda.is_available()

# Utility functions


def cast_and_place(tensor, dtype):
    if dtype.is_cuda:
        if bf.nccl_built() and bf.local_size() > torch.cuda.device_count():
            raise EnvironmentError(
                "Cannot run number of processes in one machine more than GPU device count"
                " in NCCL environment")
        return tensor.cuda(bf.rank() % torch.cuda.device_count()).type(dtype)
    return tensor.type(dtype)


def numerical_data_type():
    dtypes = [torch.FloatTensor, torch.DoubleTensor]
    if TEST_ON_GPU:
        dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
    return dtypes


def dimensions():
    return [1, 2, 3]


@pytest.fixture
def hier_setup():
    os.environ['BLUEFOG_NODES_PER_MACHINE'] = '2'
    bf.init()
    assert bf.size() % 2 == 0
    machine_size = int(bf.size() // 2)
    bf.set_machine_topology(bf.ExponentialGraph(machine_size))
    return bf.rank(), bf.size(), bf.local_rank(), bf.local_size()


def test_bluefog_local_size(hier_setup):
    _, true_size = mpi_env_rank_and_size()
    local_size = bf.local_size()
    assert local_size == min(2, true_size)


def test_bluefog_local_rank(hier_setup):
    true_rank, true_size = mpi_env_rank_and_size()
    local_rank = bf.local_rank()
    assert true_rank % min(2, true_size) == local_rank


@pytest.mark.parametrize(
    "dtype,dim",
    itertools.product(numerical_data_type(), dimensions()),
)
def test_hier_allreduce(hier_setup, dtype, dim):
    rank, size, local_rank, local_size = hier_setup
    tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
    name = "hier_local_allreduce_tensor_{}_{}".format(dim, dtype)
    tensor = cast_and_place(tensor, dtype)

    expected_value = rank - local_rank + (local_size-1) / 2
    reduced_tensor = bf.allreduce(tensor, average=True, is_hierarchical_local=True,
                                  name=name)

    assert (
        list(reduced_tensor.shape) == [23] * dim
    ), "bf.allreduce (hier_avg) produces incorrect reduced shape"
    assert (
        (reduced_tensor.data - expected_value).abs().max() < EPSILON
    ), "bf.allreduce (hier_avg) produces incorrect reduced tensor"


@pytest.mark.parametrize(
    "dtype,dim",
    itertools.product(numerical_data_type(), dimensions()),
)
def test_hier_allreduce_inplace(hier_setup, dtype, dim):
    rank, size, local_rank, local_size = hier_setup
    tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
    name = "hier_local_allreduce_inplace_tensor_{}_{}".format(dim, dtype)
    tensor = cast_and_place(tensor, dtype)

    expected_value = rank - local_rank + (local_size-1) / 2
    bf.allreduce_(tensor, average=True, is_hierarchical_local=True,
                  name=name)

    assert (
        list(tensor.shape) == [23] * dim
    ), "bf.allreduce (hier_avg) produces incorrect reduced shape"
    assert (
        (tensor.data - expected_value).abs().max() < EPSILON
    ), "bf.allreduce (hier_avg) produces incorrect reduced tensor"


@pytest.mark.parametrize(
    "dtype,dim",
    itertools.product(numerical_data_type(), dimensions()),
)
def test_hier_neighbor_allreduce(hier_setup, dtype, dim):
    rank, size, local_rank, local_size = hier_setup
    # This particular value is chosen such that the local allreduce will result in the machine rank.
    # For example, there are 8 nodes with 4 nodes per machine.
    # The allreduce result from node 0 to 3 will be 0, and the one from node 4 to 7 will 1.
    tensor = torch.FloatTensor(*([23] * dim)).fill_((rank-(local_size-1)/2.0)/local_size)
    name = "hier_neighbor_allreduce_tensor_{}_{}".format(dim, dtype)
    tensor = cast_and_place(tensor, dtype)

    reduced_tensor = bf.hierarchical_neighbor_allreduce(tensor, name=name)
    neighbor_ranks = bf.in_neighbor_machine_ranks()
    expected_value = ((rank-local_rank)/local_size + sum(neighbor_ranks))/(len(neighbor_ranks)+1)
    assert (
        list(reduced_tensor.shape) == [23] * dim
    ), "bf.hierarchical_neighbor_allreduce (hier_NA) produces incorrect reduced shape"
    assert (
        (reduced_tensor.data - expected_value).abs().max() < EPSILON
    ), "bf.hierarchical_neighbor_allreduce (hier_NA) produces incorrect reduced tensor"

@pytest.mark.parametrize(
    "dtype,dim",
    itertools.product(numerical_data_type(), dimensions()),
)
def test_hier_neighbor_allreduce_dynamic_move(hier_setup, dtype, dim):
    rank, size, local_rank, local_size = hier_setup
    machine_rank = (rank-local_rank)//local_size
    machine_size = size//local_size
    # This particular value is chosen such that the local allreduce will result in the machine rank.
    # For example, there are 8 nodes with 4 nodes per machine.
    # The allreduce result from node 0 to 3 will be 0, and the one from node 4 to 7 will 1.
    tensor = torch.FloatTensor(*([23] * dim)).fill_((rank-(local_size-1)/2.0)/local_size)
    name = "hier_neighbor_allreduce_tensor_{}_{}".format(dim, dtype)
    tensor = cast_and_place(tensor, dtype)

    reduced_tensor = bf.hierarchical_neighbor_allreduce(
        tensor, self_weight=0.0,
        src_machine_weights={(machine_rank+1)%machine_size: 1.0},
        dst_machine_weights=[(machine_rank-1)%machine_size], name=name)
    expected_value = (machine_rank+1)%machine_size
    assert (
        list(reduced_tensor.shape) == [23] * dim
    ), "bf.hierarchical_neighbor_allreduce (hier_NA) produces incorrect reduced shape"
    assert (
        (reduced_tensor.data - expected_value).abs().max() < EPSILON
    ), "bf.hierarchical_neighbor_allreduce (hier_NA) produces incorrect reduced tensor"

@pytest.mark.parametrize(
    "dtype,dim",
    itertools.product(numerical_data_type(), dimensions()),
)
def test_hier_neighbor_allreduce_dynamic_move_dst_weight(hier_setup, dtype, dim):
    rank, size, local_rank, local_size = hier_setup
    machine_rank = (rank-local_rank)//local_size
    machine_size = size//local_size
    # This particular value is chosen such that the local allreduce will result in the machine rank.
    # For example, there are 8 nodes with 4 nodes per machine.
    # The allreduce result from node 0 to 3 will be 0, and the one from node 4 to 7 will 1.
    tensor = torch.FloatTensor(*([23] * dim)).fill_((rank-(local_size-1)/2.0)/local_size)
    name = "hier_neighbor_allreduce_tensor_{}_{}".format(dim, dtype)
    tensor = cast_and_place(tensor, dtype)

    reduced_tensor = bf.hierarchical_neighbor_allreduce(
        tensor, self_weight=0.0,
        src_machine_weights={(machine_rank+1)%machine_size: 0.5},
        dst_machine_weights={(machine_rank-1)%machine_size: 2.0}, name=name)
    expected_value = (machine_rank+1)%machine_size
    assert (
        list(reduced_tensor.shape) == [23] * dim
    ), "bf.hierarchical_neighbor_allreduce (hier_NA) produces incorrect reduced shape"
    assert (
        (reduced_tensor.data - expected_value).abs().max() < EPSILON
    ), "bf.hierarchical_neighbor_allreduce (hier_NA) produces incorrect reduced tensor"

@pytest.mark.skip("Fail Github Action")
@pytest.mark.parametrize(
    "dtype,dim",
    itertools.product(numerical_data_type(), dimensions()),
)
def test_hier_neighbor_allreduce_fusion(hier_setup, dtype, dim):
    rank, size, local_rank, local_size = hier_setup
    machine_rank = (rank-local_rank)//local_size
    machine_size = size//local_size

    neighbor_ranks = bf.in_neighbor_machine_ranks()
    expected_value = (machine_rank + sum(neighbor_ranks))/(len(neighbor_ranks)+1)

    K = 50 # number of tensors send in short time
    tensor_list, handles, names = [], [], []
    for i in range(K):
        tensor = torch.FloatTensor(*([23] * dim)).fill_(i+(rank-(local_size-1)/2.0)/local_size)
        tensor = cast_and_place(tensor, dtype)
        tensor_list.append(tensor)
        names.append("index{}_{}_{}".format(i, dtype, dim))

    for i in range(K):
        handle = bf.hierarchical_neighbor_allreduce_nonblocking(
            tensor_list[i], name=names[i])
        handles.append(handle)

    outputs = []
    for i in range(K):
        output = bf.synchronize(handles[i])
        outputs.append(output)

    for i in range(K):
        assert (
            list(outputs[i].shape) == [23] * dim
        ), f"{names[i]} (hierarchical neighbor allreduce fusion) produces incorrect reduced shape"
        assert (
            (outputs[i]-expected_value-i).abs().max() < EPSILON
        ), (f"{names[i]} (hierarchical neighbor allreduce fusion) produces incorrect reduced tensor"
            f" when K = {i}")

@pytest.mark.skip("Fail Github Action")
@pytest.mark.parametrize(
    "dtype,dim",
    itertools.product(numerical_data_type(), dimensions()),
)
def test_hier_neighbor_allreduce_dynamic_move_fusion(hier_setup, dtype, dim):
    rank, size, local_rank, local_size = hier_setup
    machine_rank = (rank-local_rank)//local_size
    machine_size = size//local_size

    expected_value = (machine_rank+1)%machine_size
    src_machine_weights = {(machine_rank+1)%machine_size: 1.0}
    dst_machine_weights = [(machine_rank-1)%machine_size]

    K = 50 # number of tensors send in short time
    tensor_list, handles, names = [], [], []
    for i in range(K):
        tensor = torch.FloatTensor(*([23] * dim)).fill_(i+(rank-(local_size-1)/2.0)/local_size)
        tensor = cast_and_place(tensor, dtype)
        tensor_list.append(tensor)
        names.append("index{}_{}_{}".format(i, dtype, dim))

    for i in range(K):
        handle = bf.hierarchical_neighbor_allreduce_nonblocking(
            tensor_list[i], self_weight=0.0,
            src_machine_weights=src_machine_weights,
            dst_machine_weights=dst_machine_weights, name=names[i])
        handles.append(handle)

    outputs = []
    for i in range(K):
        output = bf.synchronize(handles[i])
        outputs.append(output)

    for i in range(K):
        assert (
            list(outputs[i].shape) == [23] * dim
        ), f"{names[i]} (hierarchical neighbor allreduce fusion) produces incorrect reduced shape"
        assert (
            (outputs[i]-expected_value-i).abs().max() < EPSILON
        ), f"{names[i]} (hierarchical neighbor allreduce fusion) produces incorrect reduced tensor"

@pytest.mark.skip("Fail Github Action")
@pytest.mark.parametrize(
    "dtype,dim",
    itertools.product(numerical_data_type(), dimensions()),
)
def test_hier_neighbor_allreduce_dynamic_move_dst_weight_fusion(hier_setup, dtype, dim):
    rank, size, local_rank, local_size = hier_setup
    machine_rank = (rank-local_rank)//local_size
    machine_size = size//local_size

    expected_value = (machine_rank+1)%machine_size
    src_machine_weights = {(machine_rank+1)%machine_size: 0.5}
    dst_machine_weights = {(machine_rank-1)%machine_size: 2.0}

    K = 50 # number of tensors send in short time
    tensor_list, handles, names = [], [], []
    for i in range(K):
        tensor = torch.FloatTensor(*([23] * dim)).fill_(i+(rank-(local_size-1)/2.0)/local_size)
        tensor = cast_and_place(tensor, dtype)
        tensor_list.append(tensor)
        names.append("index{}_{}_{}".format(i, dtype, dim))

    for i in range(K):
        handle = bf.hierarchical_neighbor_allreduce_nonblocking(
            tensor_list[i], self_weight=0.0,
            src_machine_weights=src_machine_weights,
            dst_machine_weights=dst_machine_weights, name=names[i])
        handles.append(handle)

    outputs = []
    for i in range(K):
        output = bf.synchronize(handles[i])
        outputs.append(output)

    for i in range(K):
        assert (
            list(outputs[i].shape) == [23] * dim
        ), f"{names[i]} (hierarchical neighbor allreduce fusion) produces incorrect reduced shape"
        assert (
            (outputs[i]-expected_value-i).abs().max() < EPSILON
        ), f"{names[i]} (hierarchical neighbor allreduce fusion) produces incorrect reduced tensor"
