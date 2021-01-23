# Copyright 2020 Bluefog Team. All Rights Reserved.
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

import inspect
import itertools
import time
import warnings
import unittest

from bluefog.common import topology_util
import bluefog.torch as bf
import torch
import numpy as np

EPSILON = 1e-5
TEST_ON_GPU = torch.cuda.is_available()
DIM_SIZE = 23


class WinOpsTests(unittest.TestCase):
    """
    Tests for bluefog/torch/mpi_ops.py on one-sided communication.
    """

    def __init__(self, *args, **kwargs):
        super(WinOpsTests, self).__init__(*args, **kwargs)
        warnings.simplefilter("module")

    def setUp(self):
        # Unfortunately, MPICH implementation have problem on running win ops
        # with negotiate stage as well.
        bf.init()
        bf.set_skip_negotiate_stage(True)

    def tearDown(self):
        assert bf.win_free()

    @staticmethod
    def cast_and_place(tensor, dtype):
        if dtype.is_cuda:
            if bf.nccl_built() and bf.local_size() > torch.cuda.device_count():
                raise EnvironmentError(
                    "Cannot run number of processes in one machine more than GPU device count"
                    " in NCCL environment")
            return tensor.cuda(bf.local_rank() % torch.cuda.device_count()).type(dtype)
        return tensor.type(dtype)

    def test_win_create_and_sync_and_free(self):
        """Test that the window create and free objects correctly."""
        size = bf.size()
        rank = bf.rank()
        # OpenMPI implementation seems won't allow win_create on size 1.
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return

        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_create_{}_{}".format(dim, dtype)
            is_created = bf.win_create(tensor, window_name)
            assert is_created, "bf.win_create do not create window object successfully."

            sync_result = bf.win_update(window_name)
            assert (list(sync_result.shape) == [DIM_SIZE] * dim), (
                "bf.win_update produce wrong shape tensor.")
            assert (sync_result.data.min() == rank), (
                "bf.win_update produces wrong tensor value " +
                "{0}!={1} at rank {1}.".format(sync_result.data.min(), rank))
            assert (sync_result.data.max() == rank), (
                "bf.win_update produces wrong tensor value " +
                "{0}!={1} at rank {1}.".format(sync_result.data.max(), rank))

        for dtype, dim in itertools.product(dtypes, dims):
            window_name = "win_create_{}_{}".format(dim, dtype)
            is_freed = bf.win_free(window_name)
            assert is_freed, "bf.win_free do not free window object successfully."

    def test_win_free_all(self):
        size = bf.size()
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_create_{}_{}".format(dim, dtype)
            is_created = bf.win_create(tensor, window_name)
            assert is_created, "bf.win_create do not create window object successfully."

        is_freed = bf.win_free()
        assert is_freed, "bf.win_free do not free window object successfully."

    def test_win_get_names(self):
        tensor_1 = torch.FloatTensor([1])
        tensor_2 = torch.FloatTensor([2])
        win_names = bf.get_current_created_window_names()
        assert not win_names
        assert bf.win_create(tensor_1, "1")
        win_names = bf.get_current_created_window_names()
        assert win_names == ["1"]
        assert bf.win_create(tensor_2, "2")
        win_names = bf.get_current_created_window_names()
        assert win_names == ["1", "2"]
        assert bf.win_free("1")
        win_names = bf.get_current_created_window_names()
        assert win_names == ["2"]
        assert bf.win_free()
        win_names = bf.get_current_created_window_names()
        assert not win_names

    def test_win_update_with_given_weights(self):
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_create_{}_{}".format(dim, dtype)
            is_created = bf.win_create(tensor, window_name)
            assert is_created, "bf.win_create do not create window object successfully."

            # Test simple average rule.
            weight = 1.0/(len(bf.in_neighbor_ranks())+1)
            sync_result = bf.win_update(window_name,
                                        self_weight=weight,
                                        neighbor_weights={
                                            x: weight for x in bf.in_neighbor_ranks()}
                                        )
            assert (list(sync_result.shape) == [DIM_SIZE] * dim), (
                "bf.win_update (weighted) produces wrong shape tensor.")
            assert (sync_result.data - rank).abs().max() < EPSILON, (
                "bf.win_update (weighted) produces wrong tensor value " +
                "[{0}-{1}]!={2} at rank {2}.".format(sync_result.min(),
                                                     sync_result.max(), rank))

    def test_win_update_with_default_weights(self):
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

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_create_{}_{}".format(dim, dtype)
            is_created = bf.win_create(tensor, window_name)
            assert is_created, "bf.win_create do not create window object successfully."

            # Note the buffers store the copy of original value so they will not change.
            tensor.mul_(2)
            if rank == 0:
                expected_result = rank * 2 / size + rank * (size-1)/size
            else:
                expected_result = rank / size + rank * 2 * (1-1/size)

            sync_result = bf.win_update(window_name)
            assert (list(sync_result.shape) == [DIM_SIZE] * dim), (
                "bf.win_update (weighted) produces wrong shape tensor.")
            assert (sync_result.data - expected_result).abs().max() < EPSILON, (
                "bf.win_update (weighted) produces wrong tensor value " +
                "[{0}-{1}]!={2} at rank {2}.".format(sync_result.min(),
                                                     sync_result.max(), rank))
        assert bf.win_free()

    def test_win_update_then_collect(self):
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        indegree = int(np.ceil(np.log2(size)))
        expected_result = rank * (indegree+1)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_update_collect_{}_{}".format(dim, dtype)

            bf.win_create(tensor, window_name)

            # After the collect ops, the neighbro tensor will become zero.
            # So second win_update_then_collect should produce the same value.
            for _ in range(2):
                collect_tensor = bf.win_update_then_collect(window_name)

                assert (list(collect_tensor.shape) == [DIM_SIZE] * dim), (
                    "bf.win_update_then_collect produces wrong shape tensor.")
                assert (collect_tensor.data - expected_result).abs().max() < EPSILON, (
                    "bf.win_update_then_collect produces wrong tensor value " +
                    "[{0}-{1}]!={2} at rank {2}.".format(collect_tensor.min(),
                                                         collect_tensor.max(), rank))

    def test_win_put(self):
        """Test that the window put operation."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) %
                          size for i in range(indegree)]  # in-neighbor
        avg_value = (rank + np.sum(neighbor_ranks)) / float(indegree+1)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_put_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)

            bf.win_put(tensor, window_name)
            bf.barrier()
            sync_result = bf.win_update(window_name)
            assert (list(sync_result.shape) == [DIM_SIZE] * dim), (
                "bf.win_update after win_put produces wrong shape tensor.")
            assert (sync_result.data - avg_value).abs().max() < EPSILON, (
                "bf.win_update after win_put produces wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format(sync_result.min(),
                                                 sync_result.max(), avg_value, rank))

        time.sleep(0.5)
        for dtype, dim in itertools.product(dtypes, dims):
            window_name = "win_put_{}_{}".format(dim, dtype)
            is_freed = bf.win_free(window_name)
            assert is_freed, "bf.win_free do not free window object successfully."

    def test_get_win_version_with_win_put(self):
        """Test version window is initialized, updated and cleared correctly with win put."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) %
                          size for i in range(indegree)]  # in-neighbor

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_version_put_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            original_versions = list(bf.get_win_version(window_name).values())
            bf.barrier()
            bf.win_put(tensor, window_name)
            bf.barrier()
            versions_after_win_get = list(
                bf.get_win_version(window_name).values())
            bf.win_update(window_name)
            versions_after_win_update = list(
                bf.get_win_version(window_name).values())
            neighbor_ranks_number = len(neighbor_ranks)

            zero_number_in_original_versions = len(
                original_versions) - np.count_nonzero(original_versions)
            assert (zero_number_in_original_versions == neighbor_ranks_number), (
                "version initialization is wrong.")

            zero_number_after_win_update = len(
                versions_after_win_update) - np.count_nonzero(versions_after_win_update)
            assert (zero_number_after_win_update == neighbor_ranks_number), (
                "version clear up is wrong.")

            expected_versions_after_win_get = [1] * neighbor_ranks_number

            assert (versions_after_win_get == expected_versions_after_win_get), (
                "version after win put is wrong.")

        for dtype, dim in itertools.product(dtypes, dims):
            window_name = "win_version_put_{}_{}".format(dim, dtype)
            is_freed = bf.win_free(window_name)
            assert is_freed, "bf.win_free do not free window object successfully."

    def test_win_put_with_varied_tensor_elements(self):
        """Test that the window put operation."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) %
                          size for i in range(indegree)]  # in-neighbor
        avg_value = (rank + np.sum(neighbor_ranks)) / float(indegree+1)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            base_tensor = torch.arange(
                DIM_SIZE**dim, dtype=torch.float32).view_as(tensor).div(1000)
            tensor = self.cast_and_place(tensor, dtype)
            base_tensor = self.cast_and_place(base_tensor, dtype)
            tensor = tensor + base_tensor
            window_name = "win_put_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)

            bf.win_put(tensor, window_name)
            bf.barrier()
            sync_result = bf.win_update(window_name)
            assert (list(sync_result.shape) == [DIM_SIZE] * dim), (
                "bf.win_update after win_put produces wrong shape tensor.")
            assert ((sync_result-base_tensor).data - avg_value).abs().max() < EPSILON, (
                "bf.win_update after win_put produces wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format((sync_result-base_tensor).min(),
                                                 (sync_result-base_tensor).max(), avg_value, rank))

        time.sleep(0.5)
        for dtype, dim in itertools.product(dtypes, dims):
            window_name = "win_put_{}_{}".format(dim, dtype)
            is_freed = bf.win_free(window_name)
            assert is_freed, "bf.win_free do not free window object successfully."

    def test_win_put_with_given_destination(self):
        """Test that the window put operation with given destination."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        indegree = int(np.ceil(np.log2(size)))
        # We use given destination to form a (right-)ring.
        avg_value = (rank*indegree + 1.23*((rank-1) %
                                           size)) / float(indegree+1)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_put_given_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            bf.win_put(tensor, window_name,
                       dst_weights={(rank+1) % size: 1.23})
            bf.barrier()
            sync_result = bf.win_update(window_name)
            assert (list(sync_result.shape) == [DIM_SIZE] * dim), (
                "bf.win_update after win_put given destination produces wrong shape tensor.")
            assert (sync_result.data - avg_value).abs().max() < EPSILON, (
                "bf.win_update after win_put given destination produces wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format(sync_result.min(),
                                                 sync_result.max(), avg_value, rank))

        time.sleep(0.5)
        for dtype, dim in itertools.product(dtypes, dims):
            window_name = "win_put_given_{}_{}".format(dim, dtype)
            is_freed = bf.win_free(window_name)
            assert is_freed, "bf.win_free do not free window object successfully."

    def test_win_accumulate(self):
        """Test that the window accumulate operation."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        outdegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) %
                          size for i in range(outdegree)]  # in-neighbor
        avg_value = rank + np.sum(neighbor_ranks) / float(outdegree+1)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_accumulate_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            bf.win_accumulate(tensor, window_name)

            bf.barrier()
            sync_result = bf.win_update(window_name)

            assert (list(sync_result.shape) == [DIM_SIZE] * dim), (
                "bf.win_update after win_accmulate produces wrong shape tensor.")
            assert (sync_result.data - avg_value).abs().max() < EPSILON, (
                "bf.win_update after win_accmulate produces wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format(sync_result.min(),
                                                 sync_result.max(), avg_value, rank))

    def test_win_accumulate_with_varied_tensor_elements(self):
        """Test that the window accumulate operation."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        outdegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) %
                          size for i in range(outdegree)]  # in-neighbor
        avg_value = rank + np.sum(neighbor_ranks) / float(outdegree+1)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            base_tensor = torch.arange(
                DIM_SIZE**dim, dtype=torch.float32).view_as(tensor).div(1000)
            tensor = self.cast_and_place(tensor, dtype)
            base_tensor = self.cast_and_place(base_tensor, dtype)
            tensor = tensor + base_tensor
            window_name = "win_accumulate_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            bf.win_accumulate(tensor, window_name)

            bf.barrier()
            sync_result = bf.win_update(window_name)
            sync_base_tensor = base_tensor*(1+outdegree/(outdegree+1))

            assert (list(sync_result.shape) == [DIM_SIZE] * dim), (
                "bf.win_update after win_accmulate produces wrong shape tensor.")
            assert ((sync_result-sync_base_tensor).data - avg_value).abs().max() < EPSILON, (
                "bf.win_update after win_accmulate produces wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format((sync_result-sync_base_tensor).min(),
                                                 (sync_result -
                                                  sync_base_tensor).max(),
                                                 avg_value, rank))

    def test_win_accumulate_with_given_destination(self):
        """Test that the window accumulate operation with given destination."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        avg_value = rank + ((rank-1) % size) * 1.23 / 2.0

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_accumulate_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            bf.win_accumulate(tensor, window_name,
                              dst_weights={(rank+1) % size: 1.23})

            bf.barrier()
            sync_result = bf.win_update(window_name,
                                        self_weight=0.5,
                                        neighbor_weights={(rank-1) % size: 0.5})

            assert (list(sync_result.shape) == [DIM_SIZE] * dim), (
                "bf.win_update after win_accmulate given destination produces wrong shape tensor.")
            assert (sync_result.data - avg_value).abs().max() < EPSILON, (
                "bf.win_update after win_accmulate given destination produces wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format(sync_result.min(),
                                                 sync_result.max(), avg_value, rank))

    def test_win_get(self):
        """Test that the window get operation."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) %
                          size for i in range(indegree)]  # in-neighbor
        avg_value = (rank + np.sum(neighbor_ranks)) / float(indegree+1)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_get_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            bf.win_get(window_name)
            bf.barrier()
            recv_tensor = bf.win_update(window_name, clone=True)

            assert (list(recv_tensor.shape) == [DIM_SIZE] * dim), (
                "bf.win_get produce wrong shape tensor.")
            assert (recv_tensor.data - avg_value).abs().max() < EPSILON, (
                "bf.win_get produce wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format(
                    recv_tensor.min(), recv_tensor.max(), avg_value, rank))

    def test_get_win_version_with_win_get(self):
        """Test version window is initialized, updated and cleared correctly with win get."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) %
                          size for i in range(indegree)]  # in-neighbor

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_version_get_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            original_versions = list(bf.get_win_version(window_name).values())
            bf.barrier()
            bf.win_get(window_name)
            bf.barrier()
            versions_after_win_get = list(
                bf.get_win_version(window_name).values())
            bf.win_update(window_name, clone=True)
            versions_after_win_update = list(
                bf.get_win_version(window_name).values())
            neighbor_ranks_number = len(neighbor_ranks)

            zero_number_in_original_versions = len(
                original_versions) - np.count_nonzero(original_versions)
            assert ((zero_number_in_original_versions) == neighbor_ranks_number), (
                "version initialization is wrong.")

            zero_number_after_win_update = len(
                versions_after_win_update) - np.count_nonzero(versions_after_win_update)
            assert ((zero_number_after_win_update) == neighbor_ranks_number), (
                "version clear up is wrong.")

            expected_versions_after_win_get = [1] * neighbor_ranks_number

            assert (versions_after_win_get == expected_versions_after_win_get), (
                "version after win get is wrong.")

        for dtype, dim in itertools.product(dtypes, dims):
            window_name = "win_version_get_{}_{}".format(dim, dtype)
            is_freed = bf.win_free(window_name)
            assert is_freed, "bf.win_free do not free window object successfully."

    def test_win_get_with_varied_tensor_elements(self):
        """Test that the window get operation."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # By default, we use exponential two ring topology.
        indegree = int(np.ceil(np.log2(size)))
        neighbor_ranks = [(rank - 2**i) %
                          size for i in range(indegree)]  # in-neighbor
        avg_value = (rank + np.sum(neighbor_ranks)) / float(indegree+1)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            base_tensor = torch.arange(
                DIM_SIZE**dim, dtype=torch.float32).view_as(tensor).div(1000)
            tensor = self.cast_and_place(tensor, dtype)
            base_tensor = self.cast_and_place(base_tensor, dtype)
            tensor = tensor + base_tensor
            window_name = "win_get_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            bf.win_get(window_name)
            bf.barrier()
            recv_tensor = bf.win_update(window_name, clone=True)

            assert (list(recv_tensor.shape) == [DIM_SIZE] * dim), (
                "bf.win_get produce wrong shape tensor.")
            assert ((recv_tensor - base_tensor).data - avg_value).abs().max() < EPSILON, (
                "bf.win_get produce wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format((recv_tensor-base_tensor).min(),
                                                 (recv_tensor-base_tensor).max(), avg_value, rank))

    def test_win_get_with_given_sources(self):
        """Test that the window get operation with given sources."""
        size = bf.size()
        rank = bf.rank()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        # We use given destination to form a (right-)ring.
        avg_value = (rank + 1.23*((rank-1) % size)) / float(2)

        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([DIM_SIZE] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_get_given_{}_{}".format(dim, dtype)
            bf.win_create(tensor, window_name)
            bf.win_get(window_name, src_weights={
                (rank-1) % size: 1.23})
            bf.barrier()
            recv_tensor = bf.win_update(window_name,
                                        self_weight=0.5,
                                        neighbor_weights={
                                            (rank-1) % size: 0.5},
                                        clone=True)

            assert (list(recv_tensor.shape) == [DIM_SIZE] * dim), (
                "bf.win_get with given sources produces wrong shape tensor.")
            assert (recv_tensor.data - avg_value).abs().max() < EPSILON, (
                "bf.win_get with given sources produces wrong tensor value " +
                "[{}-{}]!={} at rank {}.".format(recv_tensor.min(),
                                                 recv_tensor.max(), avg_value, rank))

    def test_win_mutex_full(self):
        size = bf.size()
        rank = bf.rank()
        if size <= 2:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn(
                "Skip {} because it only supports test over at least 3 nodes".format(fname))
            return
        bf.set_topology(topology_util.FullyConnectedGraph(size))

        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        for dtype in dtypes:
            tensor = torch.FloatTensor([DIM_SIZE]).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_mutex_full_{}".format(dtype)
            bf.win_create(tensor, window_name)

            if rank == 0:
                with bf.win_mutex(window_name, for_self=True):
                    bf.barrier()
                    time.sleep(1.01)
            else:
                bf.barrier()
                t_start = time.time()
                with bf.win_mutex(window_name):
                    time.sleep(0.001)
                t_end = time.time()
                assert (t_end - t_start) > 1, \
                    "The mutex acquire time should be longer than 1 second"
                assert (t_end - t_start) < 2, \
                    "The mutex acquire time should be shorter than 2 second"

    @unittest.skip("It is most likely because the win_mutex is called through the main thread")
    def test_win_mutex_given_ranks(self):
        size = bf.size()
        rank = bf.rank()
        if size < 4:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn(
                "Skip {} because it only supports test above 4 nodes".format(fname))
            return

        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU:
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        for dtype in dtypes:
            tensor = torch.FloatTensor([DIM_SIZE]).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_mutex_given_ranks_{}".format(dtype)
            bf.win_create(tensor, window_name)
            if rank == 0:
                with bf.win_mutex(window_name, for_self=True, ranks=[1]):
                    bf.barrier()
                    time.sleep(1.01)
            elif rank == 1:
                bf.barrier()
                t_start = time.time()
                with bf.win_mutex(window_name, ranks=[0]):
                    time.sleep(0.001)
                t_end = time.time()
                assert (t_end - t_start) > 1
            elif rank == 2:
                bf.barrier()
                t_start = time.time()
                with bf.win_mutex(window_name, ranks=[0]):
                    time.sleep(0.001)
                t_end = time.time()
                assert (t_end - t_start) < 0.1
            else:
                bf.barrier()

    def test_asscoicated_with_p(self):
        size = bf.size()
        rank = bf.rank()
        if size <= 3:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn(
                "Skip {} because it only supports test over at least 3 nodes".format(fname))
            return

        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if TEST_ON_GPU and not bf.nccl_built():
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

        bf.set_topology(topology_util.RingGraph(size))
        bf.turn_on_win_ops_with_associated_p()
        for dtype, send_rank in itertools.product(dtypes, range(size)):
            tensor = torch.FloatTensor([23]).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_asscoicate_with_p_{}_{}".format(dtype, send_rank)
            bf.win_create(tensor, window_name)
            left_neighbor_rank = (send_rank - 1) % size
            right_neighbor_rank = (send_rank + 1) % size
            if rank == send_rank:
                bf.win_accumulate(tensor, name=window_name,
                                  self_weight=0.5,
                                  dst_weights={left_neighbor_rank: 0.5,
                                               right_neighbor_rank: 0.5})
            bf.barrier()
            bf.win_update_then_collect(name=window_name)
            associated_p = bf.win_associated_p(name=window_name)
            if rank == send_rank:
                assert associated_p == 0.5, (
                    "associated_p for sender {} is wrong. Get {}".format(
                        rank, associated_p))
            elif (rank == left_neighbor_rank) or (rank == right_neighbor_rank):
                assert (associated_p - 1.5) < EPSILON, (
                    "associated_p for received neighbor {} is wrong. Get {}".format(
                        rank, associated_p))
            else:
                assert associated_p == 1.0, (
                    "associated_p for untouched node {} is wrong. Get {}".format(
                        rank, associated_p))
        bf.turn_off_win_ops_with_associated_p()

    def test_asscoicated_with_p_random_test(self):
        size = bf.size()
        rank = bf.rank()
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        # Current, nccl version hasn't supported the associated with p yet.
        if TEST_ON_GPU and not bf.nccl_built():
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
        dims = [1]
        bf.turn_on_win_ops_with_associated_p()
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([23] * dim)).fill_(1)
            tensor = self.cast_and_place(tensor, dtype)
            window_name = "win_asscoicate_with_p_random_{}_{}".format(
                dim, dtype)
            bf.win_create(tensor, window_name, zero_init=True)
            for _ in range(10):
                random_weights = np.random.rand(
                    len(bf.out_neighbor_ranks()) + 1)
                random_weights /= random_weights.sum()
                self_weight = random_weights[-1]
                dst_weights = {r: random_weights[i]
                               for i, r in enumerate(bf.out_neighbor_ranks())}
                bf.win_put(tensor, self_weight=self_weight,
                           dst_weights=dst_weights, name=window_name, require_mutex=True)
                bf.win_update(name=window_name, require_mutex=True)
                bf.win_accumulate(tensor, name=window_name, require_mutex=True,
                                  self_weight=self_weight, dst_weights=dst_weights)
                bf.win_update_then_collect(name=window_name)
            bf.barrier()
            bf.win_update_then_collect(name=window_name)
            associated_p = bf.win_associated_p(name=window_name)
            # Because the associated p should operate the same as tensor always
            # the following assert should be true no matter what order is excuted.
            assert abs(associated_p - tensor.data[0]) < EPSILON

        bf.turn_off_win_ops_with_associated_p()


if __name__ == "__main__":
    unittest.main()
