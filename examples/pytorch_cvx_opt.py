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
print("Rank {}: Normal consensus result".format(bf.rank()), x[0, 0])

# Change to star topology with hasting rule, which should be unbiased as well.
bf.set_topology(topology_util.StarGraph(bf.size()), is_weighted=True)
x = torch.FloatTensor(1000, 1000).fill_(1).mul_(bf.rank())
for i in range(50):
    x = bf.neighbor_allreduce(x, name='liuji')
    print(i, end='\r')

# Expected average should be (0+1+2+...+size-1)/(size) = (size-1)/2
print("Rank {}: consensus with weights".format(bf.rank()), x[0, 0])

# Use win_accumulate to simulate the push-sum algorithm (sync).
bf.set_topology(topology_util.PowerTwoRingGraph(bf.size()))
outdegree = len(bf.out_neighbor_ranks())
indegree = len(bf.in_neighbor_ranks())

# we append the p at the last of data.
x = torch.Tensor([bf.rank()] * 3 + [1.0])
x2 = torch.Tensor([bf.rank()] * 10000 + [1.0])

bf.win_create(x, name="x_buff", zero_init=True)
bf.win_create(x2, name="x2_buff", zero_init=True)

for i in range(100):
    handle1 = bf.win_accumulate_async(
        x, name="x_buff",
        dst_weights={rank: 1.0 / (outdegree + 1)
                     for rank in bf.out_neighbor_ranks()},
        require_mutex=True)
    handle2 = bf.win_accumulate_async(
        x2, name="x2_buff",
        dst_weights={rank: 1.0 / (outdegree + 1)
                     for rank in bf.out_neighbor_ranks()},
        require_mutex=True)
    bf.win_wait(handle1)
    bf.win_wait(handle2)

    x.div_(1+outdegree)
    x2.div_(1+outdegree)

    bf.win_sync_then_collect(name="x_buff")
    bf.win_sync_then_collect(name="x2_buff")

bf.barrier()
# Do not forget to sync at last!
x = bf.win_sync_then_collect(name="x_buff")

print("Rank {}: consensus with win ops p: {}, x: {}, x/p: {}".format(
    bf.rank(), x[-1], x[0], x[0] / x[-1]))

sum_push_sum = bf.allreduce(x[0]/x[-1], average=False)
bfprint("Total Sum of x/p", sum_push_sum)

p_push_sum = bf.allreduce(x[-1], average=True)
bfprint("Total Sum of p", p_push_sum)

bf.win_free(name="x_buff")
