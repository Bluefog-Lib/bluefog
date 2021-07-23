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
import os
import warnings
import unittest

import numpy as np
import networkx as nx
import pytest
import torch

from common import mpi_env_rank_and_size
import bluefog.torch as bf
from bluefog.torch import (
    ExponentialGraph,
    RingGraph,
    StarGraph,
    MeshGrid2DGraph,
    FullyConnectedGraph,
)
from bluefog.torch import (
    IsTopologyEquivalent,
    InferDestinationFromSourceRanks,
    InferSourceFromDestinationRanks,
)

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class BasicsTests(unittest.TestCase):
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

    def test_set_topology_fail_with_win_create(self):
        bf.init()
        size = bf.size()
        if size <= 1:
            fname = inspect.currentframe().f_code.co_name
            warnings.warn("Skip {} due to size 1".format(fname))
            return

        tensor = torch.FloatTensor([1])
        window_name = "win_create_test"
        is_created = bf.win_create(tensor, window_name)
        assert is_created, "bf.win_create do not create window object successfully."

        if size == 1:
            expected_topology = nx.from_numpy_array(
                np.array([[0.5]]), create_using=nx.DiGraph
            )
        elif size == 2:
            expected_topology = nx.from_numpy_array(
                np.array([[0, 0.2], [0.2, 0]]), create_using=nx.DiGraph
            )
        else:
            expected_topology = RingGraph(size)

        is_set = bf.set_topology(expected_topology)
        assert not is_set, "bf.set_topology do not fail due to win_create."

        topology = bf.load_topology()
        assert isinstance(topology, nx.DiGraph)
        assert IsTopologyEquivalent(topology, ExponentialGraph(size))

        is_freed = bf.win_free()
        assert is_freed, "bf.win_free do not free window object successfully."

    def test_set_and_load_topology(self):
        bf.init()
        size = bf.size()
        if size == 4:
            expected_topology = nx.DiGraph(
                np.array(
                    [
                        [1 / 3.0, 1 / 3.0, 1 / 3.0, 0.0],
                        [0.0, 1 / 3.0, 1 / 3.0, 1 / 3.0],
                        [1 / 3.0, 0.0, 1 / 3.0, 1 / 3.0],
                        [1 / 3.0, 1 / 3.0, 0.0, 1 / 3.0],
                    ]
                )
            )
        elif size == 1:
            expected_topology = nx.DiGraph(np.array([[1.0]]))
        else:
            expected_topology = ExponentialGraph(size)
        topology = bf.load_topology()
        assert isinstance(topology, nx.DiGraph)
        assert IsTopologyEquivalent(expected_topology, topology)

    def test_in_out_neighbors_expo2(self):
        bf.init()
        rank = bf.rank()
        size = bf.size()
        assert bf.set_topology(ExponentialGraph(size))
        in_neighbors = bf.in_neighbor_ranks()
        out_neighbors = bf.out_neighbor_ranks()

        degree = int(np.ceil(np.log2(size)))
        expected_in_neighbors = sorted(
            [(rank - 2 ** i) % size for i in range(degree)])
        expected_out_neighbors = sorted(
            [(rank + 2 ** i) % size for i in range(degree)])
        assert sorted(in_neighbors) == expected_in_neighbors
        assert sorted(out_neighbors) == expected_out_neighbors

    def test_in_out_neighbors_biring(self):
        bf.init()
        rank = bf.rank()
        size = bf.size()
        assert bf.set_topology(RingGraph(size))
        in_neighbors = bf.in_neighbor_ranks()
        out_neighbors = bf.out_neighbor_ranks()

        expected_in_neighbors = list(
            set(map(lambda x: x % size, [rank - 1, rank + 1])))
        expected_out_neighbors = list(
            set(map(lambda x: x % size, [rank - 1, rank + 1]))
        )

        if size <= 1:
            expected_in_neighbors = []
            expected_out_neighbors = []

        assert sorted(in_neighbors) == expected_in_neighbors
        assert sorted(out_neighbors) == expected_out_neighbors


@pytest.mark.parametrize(
    "topo_func",
    [ExponentialGraph, RingGraph, StarGraph, MeshGrid2DGraph, FullyConnectedGraph],
)
def test_infer_destination_from_source_ranks(topo_func):
    bf.init()
    size = bf.size()
    bf.set_topology(topo_func(size))
    topo = bf.load_topology()
    in_neighbors = bf.in_neighbor_ranks()
    out_neighbors = bf.out_neighbor_ranks()

    # Make the W into average rule.
    expected_W = (nx.to_numpy_array(topo) > 0).astype(float)
    expected_W /= expected_W.sum(axis=0)

    src_ranks, W = InferDestinationFromSourceRanks(
        src_ranks=in_neighbors, construct_adjacency_matrix=True
    )
    assert sorted(src_ranks) == out_neighbors
    np.testing.assert_allclose(W, expected_W)


@pytest.mark.parametrize(
    "topo_func",
    [ExponentialGraph, RingGraph, StarGraph, MeshGrid2DGraph, FullyConnectedGraph],
)
def test_infer_source_from_destination_ranks(topo_func):
    bf.init()
    size = bf.size()
    bf.set_topology(topo_func(size))
    topo = bf.load_topology()
    in_neighbors = bf.in_neighbor_ranks()
    out_neighbors = bf.out_neighbor_ranks()

    # Make the W into average rule.
    expected_W = (nx.to_numpy_array(topo) > 0).astype(float)
    expected_W /= expected_W.sum(axis=0)

    dst_ranks, W = InferSourceFromDestinationRanks(
        dst_ranks=out_neighbors, construct_adjacency_matrix=True
    )
    assert sorted(dst_ranks) == in_neighbors
    np.testing.assert_allclose(W, expected_W)


if __name__ == "__main__":
    unittest.main()
