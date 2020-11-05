import argparse
import timeit

import numpy as np
import torch

import bluefog.torch as bf
from bluefog.common import topology_util

parser = argparse.ArgumentParser(description='PyTorch One ops Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--data-size', type=int, default=100000,
                    help='the size of data.')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--internal-num-iters', type=int, default=10,
                    help='number of single ops in one benchmark step')
parser.add_argument('--virtual-topology', type=str, default="expo2",
                    help='The underlying virtual topology. Supporting options are ' +
                    '[expo2(Default), ring, mesh, star].')
parser.add_argument('--seed', type=int, default=2020, help='Seed for randomness.')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')


args = parser.parse_args()
torch.random.manual_seed(args.seed)
data = torch.randn(args.data_size)

bf.init()
if args.virtual_topology == "expo2":
    pass
elif args.virtual_topology == "expo3":
    bf.set_topology(topology_util.ExponentialGraph(bf.size(), base=3))
elif args.virtual_topology == "expo4":
    bf.set_topology(topology_util.ExponentialGraph(bf.size(), base=4))
elif args.virtual_topology == "ring":
    bf.set_topology(topology_util.RingGraph(bf.size(), connect_style=0))
elif args.virtual_topology == "mesh":
    bf.set_topology(topology_util.RingGraph(
        bf.size(), connect_style=0), is_weighted=True)
elif args.virtual_topology == "star":
    bf.set_topology(topology_util.StarGraph(bf.size()))
elif args.virtual_topology == "full":
    bf.set_topology(topology_util.FullyConnectedGraph(bf.size()))
else:
    raise ValueError("Unknown args.virtual_topology, supporting options are " +
                     "[expo2(Default), ring, mesh, star].")

def benchmark_step():
    global args, data
    for _ in range(args.internal_num_iters):
        bf.neighbor_allreduce(data)


def log(s, nl=True):
    if bf.local_rank() != 0:
        return
    print(s, end='\n' if nl else '', flush=True)


# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
enable_profiling = args.profiler & (bf.rank() == 0)

with torch.autograd.profiler.profile(enable_profiling, True) as prof:
    for x in range(args.num_iters):
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.data_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, 'CPU'))
        img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
img_secs_sum = bf.allreduce(torch.from_numpy(
    np.array(img_secs)), average=False)
img_sec_mean_all = np.mean(img_secs_sum.numpy())
img_sec_conf_all = 1.96 * np.std(img_secs_sum.numpy())
print('[%d] Img/sec per %s: %.1f +-%.1f' %
      (bf.rank(), 'CPU', img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (bf.size(), 'CPU', img_sec_mean_all, img_sec_conf_all))
