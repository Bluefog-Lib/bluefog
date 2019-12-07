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


class WinOpsTests(unittest.TestCase):
    """
    Tests for bluefog/torch/mpi_ops.py on one-sided communication.
    """

    def __init__(self, *args, **kwargs):
        super(WinOpsTests, self).__init__(*args, **kwargs)
        warnings.simplefilter("module")

    def setUp(self):
        bf.init()

    def cast_and_place(self, tensor, dtype):
        if dtype.is_cuda:
            # Not support to run on gpu yet.
            # return tensor.cuda(bf.local_rank()).type(dtype)
            return tensor.type(dtype)
        return tensor.type(dtype)

    @unittest.skip
    def test_win_create_and_sync_and_free(self):
        """Test that the window create and free objects correctly."""
        size = bf.size()
        rank = bf.rank()
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        # By default, we use power two ring topology.
        num_indegree = int(np.ceil(np.log2(size)))

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_create_{}_{}".format(dim, dtype)
            is_created = bf.win_create(tensor, window_name)
            assert is_created, "bf.win_create do not create window object successfully"

            sync_result = bf.win_sync(window_name)
            assert (list(sync_result.shape) == [23] * dim), (
                "bf.win_sync produce wrong shape tensor.")
            assert (sync_result.data.min() == rank), (
                "bf.win_sync produces wrong tensor value " +
                "{}!={} at rank {}.".format(sync_result.data.min(), num_indegree*rank, rank))
            assert (sync_result.data.max() == rank), (
                "bf.win_sync produces wrong tensor value " +
                "{}!={} at rank {}.".format(sync_result.data.max(), num_indegree*rank, rank))

        for dtype, dim in itertools.product(dtypes, dims):
            window_name = "win_create_{}_{}".format(dim, dtype)
            is_freed = bf.win_free(window_name)
            assert is_freed, "bf.win_free do not free window object successfully"

    @unittest.skip("Not finished in C implementation yet.")
    def test_win_put_blocking(self):
        """Test that the window put operation."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            warnings.warn(
                "Skip test win_put since the world size should be larger than 1!"
            )
            return
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        # By default, we use power two ring topology.
        num_outdegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) % size for i in range(num_outdegree)] # put
        avg_value = (np.sum(neighbor_ranks)) / float(num_outdegree+1)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([3] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_create_{}_{}".format(dim, dtype)
            bf.win_create(torch.zeros_like(tensor), window_name)
            bf.win_put_blocking(tensor, window_name)
            time.sleep(0.5)
            sync_result = bf.win_sync(window_name)
            assert (list(sync_result.shape) == [3] * dim), (
                "bf.win_sync after win_put produces wrong shape tensor.")
            assert (sync_result.data - avg_value).abs().max() < EPSILON, (
                "bf.win_sync after win_put produces wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format(sync_result.min(),
                                                 sync_result.max(), avg_value, rank))
            bf.win_free(window_name)

    @unittest.skip("Not implemented in C yet.")
    def test_win_get_blocking(self):
        """Test that the window get operation."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            warnings.warn(
                "Skip test win_get since the world size should be larger than 1!"
            )
            return
        dtypes = [torch.FloatTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor]

        # By default, we use power two ring topology.
        num_indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank + 2**i) % size for i in range(num_indegree)] # get
        sum_value = np.sum(neighbor_ranks)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_create_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            time.sleep(0.2)  # wait for others' finishing create win?
            recv_tensor = torch.zeros_like(tensor)
            bf.win_get_blocking(recv_tensor, window_name, average=False)
            assert (list(recv_tensor.shape) == [23] * dim), (
                "bf.win_get produce wrong shape tensor.")
            assert (recv_tensor.data - sum_value).abs().max() < EPSILON, (
                "bf.win_get produce wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format(recv_tensor.min(),
                                                 recv_tensor.max(), sum_value, rank))
            bf.win_free(window_name)

if __name__ == "__main__":
    unittest.main()
