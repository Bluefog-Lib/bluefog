from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import unittest
import time
import warnings

import numpy as np
import torch

from common import mpi_env_rank_and_size
import bluefog.torch as bf


EPSILON = 1e-5


class OpsTests(unittest.TestCase):
    """
    Tests for bluefog/torch/mpi_ops.py
    """

    def __init__(self, *args, **kwargs):
        super(OpsTests, self).__init__(*args, **kwargs)
        warnings.simplefilter("module")

    def setUp(self):
        bf.init()

    def cast_and_place(self, tensor, dtype):
        if dtype.is_cuda:
            # Not support to run on gpu yet.
            # return tensor.cuda(bf.local_rank()).type(dtype)
            return tensor.type(dtype)
        return tensor.type(dtype)

    def test_boardcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        size = bf.size()
        if size <= 1:
            warnings.warn(
                "Skip test broadcast since the world size should be larger than 1!"
            )
            return
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            torch.manual_seed(123456)
            tensor = torch.FloatTensor(*([23] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            if bf.rank() == root_rank:
                bf.broadcast(tensor, root_rank=root_rank,
                             name="test_broadcast_tensor")
            else:
                zero_tensor = torch.zeros_like(tensor)
                output = bf.broadcast(
                    zero_tensor, root_rank=root_rank, name="test_broadcast_tensor"
                )
                max_difference = output.data.sub(tensor).max()
                assert max_difference <= 1e-4

    def test_boardcast_inplace(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            warnings.warn(
                "Skip test broadcast_inplace since the world size should be larger than 1!"
            )
            return
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            torch.manual_seed(123456)
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            root_tensor = torch.FloatTensor(
                *([23] * dim)).fill_(1).mul_(root_rank)
            tensor = self.cast_and_place(tensor, dtype)
            root_tensor = self.cast_and_place(root_tensor, dtype)

            broadcasted_tensor = bf.broadcast_(
                tensor, root_rank=root_rank, name="test_broadcast_inplace_tensor"
            )

            assert (
                tensor == broadcasted_tensor
            ).min() == 1, "hvd.broadcast does not modify source tensor"
            assert (
                broadcasted_tensor == root_tensor
            ).min() == 1, "hvd.broadcast produces incorrect broadcasted tensor"

    def test_allreduce(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        size = bf.size()
        if size <= 1:
            warnings.warn(
                "Skip test allreduce since the world size should be larger than 1!"
            )
            return
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(123456)
            tensor = torch.FloatTensor(*([23] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)

            output = bf.allreduce(tensor, average=True,
                                  name="test_allreduce_tensor")
            max_difference = output.data.sub(tensor).max()
            assert max_difference <= 1e-4

    def test_allgather(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            warnings.warn(
                "Skip test allgather since the world size should be larger than 1!"
            )
            return
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            gathered = bf.allgather(tensor)

            assert list(gathered.shape) == [23 * size] + [23] * (dim - 1)

            for i in range(size):
                rank_tensor = gathered[i * 23: (i + 1) * 23]
                assert (
                    list(rank_tensor.shape) == [23] * dim
                ), "bf.allgather produces incorrect gathered shape"
                assert (
                    rank_tensor.data.min() == i
                ), "bf.allgather produces incorrect gathered tensor"
                assert (
                    rank_tensor.data.max() == i
                ), "bf.allgather produces incorrect gathered tensor"

    def test_allgather_variable_size(self):
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            warnings.warn(
                "Skip test allgather since the world size should be larger than 1!"
            )
            return
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = torch.FloatTensor(
                *([tensor_sizes[rank]] + [17] * (dim - 1))).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            gathered = bf.allgather(tensor)

            expected_size = sum(tensor_sizes)
            assert list(gathered.shape) == [expected_size] + [17] * (dim - 1)

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = gathered[sum(
                    tensor_sizes[:i]):sum(tensor_sizes[:i + 1])]
                assert list(rank_tensor.shape) == rank_size
                assert rank_tensor.data.min() == i
                assert rank_tensor.data.max() == i

    def test_neighbor_allreduce(self):
        """Test that the neighbor all reduce 1D, 2D, 3D tensors correctly."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            warnings.warn(
                "Skip test neighbor allreduce since the world size should be larger than 1!"
            )
            return
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        # By default, we use power two ring topology.
        num_indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) % size for i in range(num_indegree)]
        sum_value = np.sum(neighbor_ranks) + rank

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            reduced_tensor = bf.neighbor_allreduce(tensor, average=True)
            assert (
                list(reduced_tensor.shape) == [23] * dim
            ), "bf.neighbor_allreduce (avg) produces incorrect reduced shape"
            assert (
                (reduced_tensor.data.mul_(num_indegree+1) -
                 sum_value).abs().max() < EPSILON
            ), "bf.neighbor_allreduce (avg) produces incorrect reduced tensor"

            reduced_tensor = bf.neighbor_allreduce(tensor, average=False)
            assert (
                list(reduced_tensor.shape) == [23] * dim
            ), "bf.neighbor_allreduce (sum) produces incorrect reduced shape"
            assert (
                reduced_tensor.data.min() == sum_value
            ), "bf.neighbor_allreduce (sum) produces incorrect reduced tensor"
            assert (
                reduced_tensor.data.max() == sum_value
            ), "bf.neighbor_allreduce (sum) produces incorrect reduced tensor"

    def test_neighbor_allgather(self):
        """Test that the neighbor all gather 1D, 2D, 3D tensors correctly."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            warnings.warn(
                "Skip test neighbor allgather since the world size should be larger than 1!"
            )
            return
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        # By default, we use power two ring topology.
        num_indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) % size for i in range(num_indegree)]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            gathered = bf.neighbor_allgather(tensor)

            assert list(gathered.shape) == [
                23 * num_indegree] + [23] * (dim - 1)

            candidate_ranks = map(float, sorted(neighbor_ranks[:]))
            gathered_ranks = []

            for i, _ in enumerate(neighbor_ranks):
                rank_tensor = gathered[i * 23: (i + 1) * 23]
                assert (
                    list(rank_tensor.shape) == [23] * dim
                ), "bf.neighbor_allgather produces incorrect gathered shape"
                assert (
                    rank_tensor.data.min() == rank_tensor.data.max()
                ), "bf.neighbor_allgather produces incorrect gathered tensor"
                gathered_ranks.append(rank_tensor.data.max().item())

            assert sorted(candidate_ranks) == gathered_ranks, \
                "bf.neighbor_allgather produces incorrect gathered tensor"


if __name__ == "__main__":
    unittest.main()
