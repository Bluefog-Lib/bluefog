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
bf.set_topology(topology_util.StartGraph(bf.size()), is_weighted=True)
x = torch.Tensor([[bf.rank()]])
for i in range(100):
    x = bf.neighbor_allreduce(x)

# Expected average should be (0+1+2+...+size-1)/(size) = (size-1)/2
print("Rank {}: consensus with weights".format(bf.rank()), x)

# Use win_put to simulate the push-sum algorithm.
bf.set_topology(topology_util.PowerTwoRingGraph(bf.size()))

p = torch.Tensor([[1/bf.size()]])
x = torch.Tensor([[bf.rank()]])
bf.win_create(x, name="x_buff")
bf.win_create(p, name="p_buff")

bf.barrier()
for i in range(200):
    bf.win_put(x, name="x_buff")
    bf.barrier()
    x = bf.win_sync(name="x_buff")
    bf.barrier()
print("Rank {}: consensus with win ops".format(bf.rank()), x)
