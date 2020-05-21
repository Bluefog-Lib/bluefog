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

import os
import time
import threading
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

    @classmethod
    def setUpClass(cls):
        cls.temp_file = './timeline_temp'

        with env(BLUEFOG_TIMELINE=cls.temp_file):
            bf.init()

    @classmethod
    def tearDownClass(cls):
        # file_name is just a temprary file generated in bluefog timeline
        file_name = f"{cls.temp_file}{bf.rank()}.json"
        os.remove(file_name)

    def test_timeline_neighbor_allreduce(self):
        x = torch.FloatTensor(10, 10).fill_(1).mul_(bf.rank())
        x = bf.neighbor_allreduce(x, name='test_neighbor_allreduce')
        time.sleep(0.1)

        file_name = f"{self.temp_file}{bf.rank()}.json"
        with open(file_name, 'r') as tf:
            timeline_text = tf.read()
            assert 'MPI_NEIGHBOR_ALLREDUCE' in timeline_text, timeline_text
            assert 'ENQUEUE_NEIGHBOR_ALLREDUCE' in timeline_text, timeline_text

    def test_timeline_neighbor_allgather(self):
        x = torch.FloatTensor(10, 10).fill_(1).mul_(bf.rank())
        x = bf.neighbor_allgather(x, name='test_neighbor_allgather')
        time.sleep(0.1)

        file_name = f"{self.temp_file}{bf.rank()}.json"
        with open(file_name, 'r') as tf:
            timeline_text = tf.read()
            assert 'MPI_NEIGHBOR_ALLGATHER' in timeline_text, timeline_text
            assert 'ENQUEUE_NEIGHBOR_ALLGATHER' in timeline_text, timeline_text

    def test_timeline_with_python_interface(self):
        bf.timeline_start_activity("test_python_interface_x", "FAKE_ACTIVITY")
        time.sleep(0.1)
        bf.timeline_end_activity("test_python_interface_x")
        time.sleep(0.1)

        file_name = f"{self.temp_file}{bf.rank()}.json"
        with open(file_name, 'r') as tf:
            timeline_text = tf.read()
            assert 'FAKE_ACTIVITY' in timeline_text, timeline_text

    def test_timeline_multi_threads(self):
        def f():
            bf.timeline_start_activity("test_multi_thread", "THREAD")
            time.sleep(0.1)
            bf.timeline_end_activity("test_multi_thread")

        t1 = threading.Thread(target=f)
        t2 = threading.Thread(target=f)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        file_name = f"{self.temp_file}{bf.rank()}.json"
        with open(file_name, 'r') as tf:
            timeline_text = tf.read()
            assert '"tid": 1' in timeline_text, timeline_text
            assert '"tid": 2' in timeline_text, timeline_text

    def test_timeline_push_sum(self):
        # Use win_accumulate to simulate the push-sum algorithm (sync).
        outdegree = len(bf.out_neighbor_ranks())
        indegree = len(bf.in_neighbor_ranks())
        # we append the p at the last of data.
        x = torch.Tensor([bf.rank()/(indegree+1), 1.0/bf.size()/(indegree+1)])

        # Remember we do not create buffer with 0.
        bf.win_create(x, name="x_buff")
        x = bf.win_update_then_collect(name="x_buff")

        for _ in range(10):
            bf.win_accumulate(
                x, name="x_buff",
                dst_weights={rank: 1.0 / (outdegree + 1)
                             for rank in bf.out_neighbor_ranks()},
                require_mutex=True)
            x.div_(1+outdegree)
            x = bf.win_update_then_collect(name="x_buff")

        bf.barrier()
        # Do not forget to sync at last!
        x = bf.win_update_then_collect(name="x_buff")

        file_name = f"{self.temp_file}{bf.rank()}.json"
        with open(file_name, 'r') as tf:
            timeline_text = tf.read()
            assert 'MPI_WIN_ACCUMULATE' in timeline_text, timeline_text
            assert 'ENQUEUE_WIN_ACCUMULATE' in timeline_text, timeline_text

        bf.win_free()


if __name__ == "__main__":
    unittest.main()
