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
        return tensor.cuda(bf.local_rank() % torch.cuda.device_count()).type(dtype)
    return tensor.type(dtype)


def numerical_date_type():
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
    itertools.product(numerical_date_type(), dimensions()),
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
    itertools.product(numerical_date_type(), dimensions()),
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
    itertools.product(numerical_date_type(), dimensions()),
)
def test_hier_neighbor_allreduce(hier_setup, dtype, dim):
    rank, size, local_rank, local_size = hier_setup
    tensor = torch.FloatTensor(*([3] * dim)).fill_(1).mul_(rank)
    name = "hier_neighbor_allreduce_tensor_{}_{}".format(dim, dtype)
    tensor = cast_and_place(tensor, dtype)

    # TODO(hhb): add real test after the hierarchical is fixed.
    pass
