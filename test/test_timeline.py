from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import torch
import unittest
import warnings

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

    def test_timeline(self):
        with tempfile.NamedTemporaryFile() as t:
            with env(BLUEFOG_TIMELINE=t.name):
                bf.init()

                x = torch.FloatTensor(10, 10).fill_(1).mul_(bf.rank())
                for i in range(50):
                    x = bf.neighbor_allreduce(x, name='test_name_1')
                    print(i, end='\r')

                print("Rank {}: Normal consensus result".format(bf.rank()),x[0,0])

                # Change to star topology with hasting rule, which should be unbiased as well.
                bf.set_topology(topology_util.StarGraph(bf.size()), is_weighted=True)
                x = torch.FloatTensor(10, 10).fill_(1).mul_(bf.rank())
                for i in range(50):
                    x = bf.neighbor_allreduce(x, name='test_name_2')
                    print(i, end='\r')

                # Expected average should be (0+1+2+...+size-1)/(size) = (size-1)/2
                print("Rank {}: consensus with weights".format(bf.rank()), x[0,0])

                if bf.rank() == 0:

                    file_name = t.name + '0.json'
                    with open(file_name, 'r') as tf:
                        timeline_text = tf.read()
                        assert 'MPI_NEIGHBOR_ALLREDUCE' in timeline_text, timeline_text
                        assert 'ENQUEUE_NEIGHBOR_ALLREDUCE' in timeline_text, timeline_text

if __name__ == "__main__":
   unittest.main()