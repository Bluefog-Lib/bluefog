from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import warnings
import numpy as np
import networkx as nx
import pytest

from common import mpi_env_rank_and_size
import bluefog.torch as bf
from bluefog.common.topology_util import PowerTwoRingGraph, BiRingGraph

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

    @unittest.skip("Annoying numpy.ufunc warnings.")
    def test_topology(self):
        _, size = mpi_env_rank_and_size()
        if size == 4:
            expected_topology = nx.DiGraph(np.array(
                [[0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0]]
            ))
        elif size == 1:
            expected_topology = nx.DiGraph(np.array([[0]]))
        else:
            expected_topology = PowerTwoRingGraph(size)
        bf.init()
        actual_topology = bf.load_topology()
        assert isinstance(actual_topology, nx.DiGraph)
        np.testing.assert_array_equal(
            nx.to_numpy_array(expected_topology), nx.to_numpy_array(actual_topology))

    @unittest.skip("Not implemented in C yet.")
    def test_set_and_load_topology(self):
        _, size = mpi_env_rank_and_size()
        ring_graph = BiRingGraph(size)
        bf.set_topology(ring_graph)
        returned_graph = bf.load_topology()

        assert isinstance(returned_graph, nx.DiGraph)

        np.testing.assert_array_equal(
            nx.to_numpy_array(ring_graph), nx.to_numpy_array(returned_graph))


if __name__ == "__main__":
    unittest.main()
