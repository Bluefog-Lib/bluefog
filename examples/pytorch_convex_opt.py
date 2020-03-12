""" Run basic consensus algorithm.

mpirun -np 16 --oversubscribe python pytorch_convex_opt.py
"""
import torch
import numpy as np
import bluefog.torch as bf
from bluefog.common import topology_util


def bfprint(*args, **kwargs):
    if bf.rank() == 0:
        print(*args, **kwargs)


bf.init()

x = torch.Tensor([[bf.rank()]])
for i in range(100):
    x = bf.neighbor_allreduce(x)

# Expected average should be (0+1+2+...+size-1)/(size) = (size-1)/2
print("Rank {}: Normal consensus result".format(bf.rank()), x)

# Change to star topology with hasting rule, which should be unbiased as well.
bf.set_topology(topology_util.StarGraph(bf.size()), is_weighted=True)
x = torch.Tensor([[bf.rank()]])
for i in range(100):
    x = bf.neighbor_allreduce(x)

# Expected average should be (0+1+2+...+size-1)/(size) = (size-1)/2
print("Rank {}: consensus with weights".format(bf.rank()), x)

# Use win_accumulate to simulate the push-sum algorithm (sync).
bf.set_topology(topology_util.PowerTwoRingGraph(bf.size()))
outdegree = len(bf.out_neighbor_ranks())
indegree = len(bf.in_neighbor_ranks())

# Remember we do not create buffer with 0.
# we append the p at the last of data.
x = torch.Tensor([bf.rank()/(indegree+1), 1.0/bf.size()/(indegree+1)])
bf.win_create(x, name="x_buff")
x = bf.win_sync_then_collect(name="x_buff")

bf.barrier()
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

sum_push_sum = bf.allreduce(x[0]/x[1], average=False)
if bf.rank() == 0:
    print("Total Sum ", sum_push_sum)

p_push_sum = bf.allreduce(x[1], average=False)
if bf.rank() == 0:
    print("Total Sum ", p_push_sum)

bf.win_free(name="x_buff")
