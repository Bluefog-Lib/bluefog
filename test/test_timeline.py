from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import unittest
import warnings

import torch
import bluefog.torch as bf
from bluefog.common import topology_util
from bluefog.common.util import env


class TimelineTests(unittest.TestCase):
    """
    Tests for timeline
    """

    def __init__(self, *args, **kwargs):
        super(TimelineTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_timeline_neighbor_allreduce(self):
        with tempfile.NamedTemporaryFile() as t, env(BLUEFOG_TIMELINE=t.name):
            bf.init()

            x = torch.FloatTensor(10, 10).fill_(1).mul_(bf.rank())
            for _ in range(50):
                x = bf.neighbor_allreduce(x, name='test_neighbor_allreduce')

            file_name = f"{t.name}{bf.rank()}.json"
            with open(file_name, 'r') as tf:
                timeline_text = tf.read()
                assert 'MPI_NEIGHBOR_ALLREDUCE' in timeline_text, timeline_text
                assert 'ENQUEUE_NEIGHBOR_ALLREDUCE' in timeline_text, timeline_text


if __name__ == "__main__":
    unittest.main()
