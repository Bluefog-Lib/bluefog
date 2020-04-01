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

import torch
import numpy as np
import bluefog.torch as bf
from bluefog.common import topology_util


def bfprint(*args, **kwargs):
    if bf.rank() == 0:
        print(*args, **kwargs)


bf.init()

x = torch.FloatTensor(1000, 1000).fill_(1).mul_(bf.rank())
for i in range(50):
    x = bf.neighbor_allreduce(x, name='ybc')
    print(i, end='\r')

# Expected average should be (0+1+2+...+size-1)/(size) = (size-1)/2
print("Rank {}: Normal consensus result".format(bf.rank()),x[0,0])

# Change to star topology with hasting rule, which should be unbiased as well.
bf.set_topology(topology_util.StarGraph(bf.size()), is_weighted=True)
x = torch.FloatTensor(1000, 1000).fill_(1).mul_(bf.rank())
for i in range(50):
    x = bf.neighbor_allreduce(x, name='liuji')
    print(i, end='\r')

# Expected average should be (0+1+2+...+size-1)/(size) = (size-1)/2
print("Rank {}: consensus with weights".format(bf.rank()), x[0,0])

# Use win_accumulate to simulate the push-sum algorithm (sync).
bf.set_topology(topology_util.PowerTwoRingGraph(bf.size()))
outdegree = len(bf.out_neighbor_ranks())
indegree = len(bf.in_neighbor_ranks())

# we append the p at the last of data.
x = torch.Tensor([bf.rank()/(indegree+1)] * 100000 + [1.0/bf.size()/(indegree+1)])

# Remember we do not create buffer with 0.
bf.win_create(x, name="x_buff")
x = bf.win_sync_then_collect(name="x_buff")

for i in range(100):
    bf.win_accumulate(
        x, name="x_buff",
        dst_weights={rank: 1.0 / (outdegree + 1) for rank in bf.out_neighbor_ranks()},
        require_mutex=True)
    x.div_(1+outdegree)
    x = bf.win_sync_then_collect(name="x_buff")

bf.barrier()
# Do not forget to sync at last!
x = bf.win_sync_then_collect(name="x_buff")

print("Rank {}: consensus with win ops p: {}, x: {}, x/p: {}".format(bf.rank(), x[1], x[0], x[0] / x[1]))

sum_push_sum = bf.allreduce(x[0]/x[-1], average=False)
if bf.rank() == 0:
    print("Total Sum ", sum_push_sum)

p_push_sum = bf.allreduce(x[-1], average=False)
if bf.rank() == 0:
    print("Total Sum ", p_push_sum)

bf.win_free(name="x_buff")
