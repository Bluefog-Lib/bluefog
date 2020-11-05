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

import unittest
import warnings
import numpy as np
import networkx as nx
import pytest
import tensorflow as tf

from common import mpi_env_rank_and_size
import bluefog.tensorflow as bf
from bluefog.common.topology_util import ExponentialGraph, RingGraph

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class BasicsTests(tf.test.TestCase):
    """
    Tests for basics.py
    """

    def __init__(self, *args, **kwargs):
        super(BasicsTests, self).__init__(*args, **kwargs)
        warnings.simplefilter("module")

    def test_bluefog_rank(self):
        """Test that the rank returned by bf.rank() is correct."""
        true_rank, _ = mpi_env_rank_and_size()
        bf.init()
        rank = bf.rank()
        # print("Rank: ", true_rank, rank)
        assert true_rank == rank

    def test_bluefog_size(self):
        """Test that the size returned by bf.size() is correct."""
        _, true_size = mpi_env_rank_and_size()
        bf.init()
        size = bf.size()
        # print("Size: ", true_size, size)
        assert true_size == size

    def test_set_and_load_topology(self):
        _, size = mpi_env_rank_and_size()
        if size == 4:
            expected_topology = nx.DiGraph(np.array(
                [[1/3., 1/3., 1/3., 0.], [0., 1/3., 1/3., 1/3.],
                 [1/3., 0., 1/3., 1/3.], [1/3., 1/3., 0., 1/3.]]
            ))
        elif size == 1:
            expected_topology = nx.DiGraph(np.array([[1.0]]))
        else:
            expected_topology = ExponentialGraph(size)
        bf.init()
        topology = bf.load_topology()
        assert isinstance(topology, nx.DiGraph)
        np.testing.assert_array_equal(
            nx.to_numpy_array(expected_topology), nx.to_numpy_array(topology))

    def test_in_out_neighbors_expo2(self):
        rank, size = mpi_env_rank_and_size()
        bf.init()
        bf.set_topology(ExponentialGraph(size))
        in_neighobrs = bf.in_neighbor_ranks()
        out_neighbors = bf.out_neighbor_ranks()

        degree = int(np.ceil(np.log2(size)))
        expected_in_neighbors = sorted([(rank - 2**i) %
                                        size for i in range(degree)])
        expected_out_neighbors = sorted([(rank + 2**i) %
                                         size for i in range(degree)])
        assert sorted(in_neighobrs) == expected_in_neighbors
        assert sorted(out_neighbors) == expected_out_neighbors

    def test_in_out_neighbors_biring(self):
        rank, size = mpi_env_rank_and_size()
        bf.init()
        bf.set_topology(RingGraph(size))
        in_neighobrs = bf.in_neighbor_ranks()
        out_neighbors = bf.out_neighbor_ranks()

        expected_in_neighbors = list(set(
            map(lambda x: x % size, [rank - 1, rank + 1])))
        expected_out_neighbors = list(set(
            map(lambda x: x % size, [rank - 1, rank + 1])))

        if size <= 1:
            expected_in_neighbors = []
            expected_out_neighbors = []

        assert sorted(in_neighobrs) == expected_in_neighbors
        assert sorted(out_neighbors) == expected_out_neighbors


if __name__ == "__main__":
    tf.test.main()
