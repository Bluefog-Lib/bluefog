from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import itertools
import unittest
import warnings

import numpy as np
import torch
import networkx as nx

import bluefog.torch as bf
from bluefog.common import topology_util


EPSILON = 1e-5
TEST_ON_GPU = torch.cuda.is_available()

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
            device_id = bf.local_rank() % torch.cuda.device_count()
            return tensor.cuda(device_id).type(dtype)
        return tensor.type(dtype)

    def test_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        size = bf.size()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.IntTensor, torch.DoubleTensor, torch.LongTensor,
                  torch.ByteTensor, torch.CharTensor, torch.ShortTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            torch.manual_seed(123456)
            tensor = torch.FloatTensor(*([23] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            name = "broadcast_tensor_{}_{}".format(dim, dtype)
            if bf.rank() == root_rank:
                bf.broadcast(tensor, root_rank=root_rank, name=name)
            else:
                zero_tensor = torch.zeros_like(tensor)
                output = bf.broadcast(
                    zero_tensor, root_rank=root_rank, name=name
                )
                max_difference = output.data.sub(tensor).max()
                assert max_difference <= 1e-4

    def test_broadcast_inplace(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.IntTensor, torch.DoubleTensor, torch.LongTensor,
                  torch.ByteTensor, torch.CharTensor, torch.ShortTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            torch.manual_seed(123456)
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            name = "broadcast_inplace_tensor_{}_{}".format(dim, dtype)
            root_tensor = torch.FloatTensor(
                *([23] * dim)).fill_(1).mul_(root_rank)
            tensor = self.cast_and_place(tensor, dtype)
            root_tensor = self.cast_and_place(root_tensor, dtype)

            broadcasted_tensor = bf.broadcast_(tensor, root_rank=root_rank, name=name)

            assert (
                tensor == broadcasted_tensor
            ).min() == 1, "bf.broadcast_ does not modify source tensor"
            assert (
                broadcasted_tensor == root_tensor
            ).min() == 1, "bf.broadcast_ produces incorrect broadcasted tensor"

    def test_allreduce_avg(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        size = bf.size()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(123456)
            tensor = torch.FloatTensor(*([23] * dim)).random_(-100, 100)
            name = "allreduce_tensor_{}_{}".format(dim, dtype)
            tensor = self.cast_and_place(tensor, dtype)

            output = bf.allreduce(tensor, average=True, name=name)
            max_difference = output.data.sub(tensor).max()
            assert max_difference <= 1e-4, "bf.allreduce(avg) produces incorrect tensor"

    def test_allreduce_sum(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        size = bf.size()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor, torch.IntTensor, torch.DoubleTensor,]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(123456)
            tensor = torch.FloatTensor(*([23] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            name = "allreduce_tensor_{}_{}".format(dim, dtype)

            output = bf.allreduce(tensor, average=False, name=name)
            max_difference = output.data.sub(tensor.mul(size)).max()
            assert max_difference <= 1e-4, "bf.allreduce(sum) produces incorrect tensor"

    def test_allgather(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.IntTensor, torch.DoubleTensor, torch.LongTensor,
                  torch.ByteTensor, torch.CharTensor, torch.ShortTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([2] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            name = "allgather_tensor_{}_{}".format(dim, dtype)
            gathered = bf.allgather(tensor, name=name)

            assert list(gathered.shape) == [2 * size] + [2] * (dim - 1)

            for i in range(size):
                rank_tensor = gathered[i * 2: (i + 1) * 2]
                assert (
                    list(rank_tensor.shape) == [2] * dim
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
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.IntTensor, torch.DoubleTensor, torch.LongTensor,
                  torch.ByteTensor, torch.CharTensor, torch.ShortTensor]
        if TEST_ON_GPU:
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
            name = "allgather_tensor_{}_{}".format(dim, dtype)
            gathered = bf.allgather(tensor, name=name)

            expected_size = sum(tensor_sizes)
            assert list(gathered.shape) == [expected_size] + [17] * (dim - 1)

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = gathered[sum(
                    tensor_sizes[:i]):sum(tensor_sizes[:i + 1])]
                assert list(rank_tensor.shape) == rank_size, \
                    "bf.allgather(var) produces incorrect gathered shape"
                assert rank_tensor.data.min() == i, \
                    "bf.allgather(var) produces incorrect gathered tensor"
                assert rank_tensor.data.max() == i, \
                    "bf.allgather(var) produces incorrect gathered tensor"

    def test_neighbor_allreduce_avg(self):
        """Test that the neighbor all reduce (avg) 1D, 2D, 3D tensors correctly."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        # By default, we use power two ring topology.
        num_indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) % size for i in range(num_indegree)]
        sum_value = np.sum(neighbor_ranks) + rank

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            name = "neighbor_allreduce_{}_{}".format(dim, dtype)
            reduced_tensor = bf.neighbor_allreduce(tensor, average=True, name=name)
            assert (
                list(reduced_tensor.shape) == [23] * dim
            ), "bf.neighbor_allreduce (avg) produces incorrect reduced shape"
            assert (
                (reduced_tensor.data.mul_(num_indegree+1) -
                 sum_value).abs().max() < EPSILON
            ), "bf.neighbor_allreduce (avg) produces incorrect reduced tensor"

    def test_neighbor_allreduce_avg_meshgrid_topo(self):
        """
        Test that the neighbor all reduce (avg) 1D, 2D, 3D tensors
        correctly in a 2D meshgrid topology.
        """
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        is_set = bf.set_topology(topology_util.MeshGrid2DGraph(size))
        assert is_set, "Topology set failed."

        topology = bf.load_topology()
        neighbor_array_with_self = np.nonzero(
            nx.to_numpy_matrix(topology)[rank])[1]
        num_indegree = len(neighbor_array_with_self)-1
        sum_value = neighbor_array_with_self.sum()

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

    def test_neighbor_allreduce_avg_biring_topo(self):
        """
        Test that the neighbor all reduce (avg) 1D, 2D, 3D tensors correctly
        in a bidirectional ring topology.
        """
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        is_set = bf.set_topology(topology_util.BiRingGraph(size))
        assert is_set, "Topology set failed."

        if size > 2:
            num_indegree = 2
            sum_value = rank+(rank+1) % size+(rank-1) % size
        else:
            num_indegree = 1
            sum_value = 1

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

    def test_neighbor_allreduce_avg_ring_topo(self):
        """
        Test that the neighbor all reduce (avg) 1D, 2D, 3D tensors correctly
        in a ring topology.
        """
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        for connect_direction in [False, True]:
            is_set = bf.set_topology(
                topology_util.RingGraph(size, connect_direction))
            assert is_set, "Topology set failed."

            num_indegree = 1
            sum_value = rank+(rank+(1 if connect_direction else -1)) % size

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

    def test_neighbor_allreduce_avg_star_topo(self):
        """
        Test that the neighbor all reduce (avg) 1D, 2D, 3D tensors correctly
        in a star topology.
        """
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        for center_rank in range(size):
            is_set = bf.set_topology(
                topology_util.StarGraph(size, center_rank))
            assert is_set, "Topology set failed."

            if rank == center_rank:
                num_indegree = size-1
                sum_value = size*(size-1)/2
            else:
                num_indegree = 1
                sum_value = rank+center_rank

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

    def test_neighbor_allreduce_sum(self):
        """Test that the neighbor all reduce (sum) 1D, 2D, 3D tensors correctly."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor,
                  torch.IntTensor, torch.LongTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        # By default, we use power two ring topology.
        num_indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) % size for i in range(num_indegree)]
        sum_value = np.sum(neighbor_ranks) + rank

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
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

    def test_neighbor_allreduce_weighted_avg(self):
        """Test that the neighbor all reduce (avg) 1D, 2D, 3D tensors correctly."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor]

        bf.set_topology(topology_util.StarGraph(size), is_weighted=True)

        if rank == 0:
            expect_result = (size-1) / 2
        else:
            expect_result = 0 * (1/size) + rank * (1-1/size)
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            reduced_tensor = bf.neighbor_allreduce(tensor, average=True)
            assert (
                list(reduced_tensor.shape) == [23] * dim
            ), "bf.neighbor_allreduce (weighted_avg) produces incorrect reduced shape"
            assert (
                (reduced_tensor.data - expect_result).abs().max() < EPSILON
            ), "bf.neighbor_allreduce (weighted_avg) produces incorrect reduced tensor"


    def test_neighbor_allgather(self):
        """Test that the neighbor all gather 1D, 2D, 3D tensors correctly."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.IntTensor, torch.DoubleTensor, torch.LongTensor,
                  torch.ByteTensor, torch.CharTensor, torch.ShortTensor]
        if TEST_ON_GPU:
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
