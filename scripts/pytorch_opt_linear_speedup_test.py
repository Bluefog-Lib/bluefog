import argparse
import timeit
import time as systemtime

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import bluefog.torch as bf
from bluefog.common import topology_util

# Parser
parser = argparse.ArgumentParser(
    description="PyTorch ImageNet Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--save-plot-file", default='plot.png', help="Saving the plot in the file."
)
parser.add_argument(
    "--plot-interactive", action='store_true', help="Use plt.show() to present the plot."
)
parser.add_argument(
    "--max-iter", action='store', type=int, default=1000, help="maximum iteration number."
)
parser.add_argument(
    "--lr", action='store', type=float, default=1e-1, help="learning rate"
)
parser.add_argument(
    "--method", help="this example supports exact_diffusion, gradient_tracking, and push_diging",
    default='exact_diffusion'
)
parser.add_argument(
    '--virtual-topology', type=str, default="expo2",
    help='The underlying virtual topology. Supporting options are ' +
    '[expo2(Default), ring, mesh, star].')

parser.add_argument(
    '--computation-mode', type=str, default="normal",
    help='Supporting modes are ' +
                    '[normal, compute_and_no_communicate, sleep_and_communicate].')

parser.add_argument(
    "--sleep-time", action='store', type=float, default=1e-1, help="sleep time"
)

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-autograd', action='store_true', default=False,
                    help='calculate gradient by hand')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument(
    "--task", help="this example supports linear_regression and logistic_regression",
    default='logistic_regression')
parser.add_argument('--data-size', type=int, default=2000,
                    help='input data size')
parser.add_argument('--data-dim', type=int, default=500,
                    help='input data dimension')            

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

bf.init()

if args.cuda:
    torch.cuda.set_device(bf.local_rank())
    cudnn.benchmark = True

def logistic_loss_step(x_, rho, X, y, tensor_name, calculate_by_hand=True):
    """Calculate gradient of logistic loss via pytorch autograd."""

    if calculate_by_hand:
        # prob = torch.exp( -y * X.mm(x_.data))
        prob = torch.exp( -y * torch.matmul(X, x_.data))
        alpha = prob / (1+prob)
        x_.grad = rho * x_.data - torch.mean(alpha*y*X, dim = 0).reshape(-1,1)
        return

    else:
        if os.getenv('BLUEFOG_TIMELINE'):
            with bf.timeline_context(tensor_name=tensor_name,
                                    activity_name="gradient computation"):
                loss_ = torch.mean(torch.log(1 + torch.exp(-y*X.mm(x_)))) + \
                    0.5*rho*torch.norm(x_, p=2)
                loss_.backward()
        else:
            loss_ = torch.mean(torch.log(1 + torch.exp(-y*X.mm(x_)))) + \
                0.5*rho*torch.norm(x_, p=2)
            loss_.backward()
        return

if args.virtual_topology == "expo2":
    pass
elif args.virtual_topology == "ring":
        bf.set_topology(topology_util.RingGraph(bf.size(), connect_style=0))
elif args.virtual_topology == "mesh":
    bf.set_topology(topology_util.RingGraph(
        bf.size(), connect_style=0), is_weighted=True)
elif args.virtual_topology == "star":
    bf.set_topology(topology_util.StarGraph(bf.size()))
else:
    raise ValueError("Unknown args.virtual_topology, supporting options are " +
                     "[expo2(Default), ring, mesh, star].")

# Set up fake data
# Generate data for logistic regression (synthesized data)
torch.random.manual_seed(123417 * bf.rank())
m, n = args.data_size, args.data_dim
X = torch.randn(m, n).to(torch.double)
w_0 = (torch.randn(n, 1)).to(torch.double)
y = torch.rand(m, 1).to(torch.double) < 1 / (1+torch.exp(X.mm(w_0)))
y = y.double()
y = 2*y - 1
rho = 1e-2

w = torch.zeros(n, 1, dtype=torch.double, requires_grad=True)

if args.cuda:
    X, y = X.cuda(), y.cuda()
    w = torch.zeros(n, 1, dtype=torch.double, requires_grad=True, device="cuda")

logistic_loss_step(
w, rho, X, y, tensor_name='neighbor.allreduce.Grad.Tracking.w', calculate_by_hand=args.no_autograd)
q = w.grad.data.clone()  # q^0 = grad(w^0)
w.grad.data.zero_()
grad_prev = q.clone()
alpha = 1e-1

def benchmark_step():

    global w, q, grad_prev, alpha

    if args.computation_mode == "normal":
        w_handle = bf.neighbor_allreduce_nonblocking(w.data, name='Grad.Tracking.w')
        w.data = - alpha * q + bf.synchronize(w_handle) 
        q_handle = bf.neighbor_allreduce_nonblocking(q, name='Grad.Tracking.q')

        # calculate local gradient
        logistic_loss_step(
            w, rho, X, y, tensor_name='neighbor.allreduce.Grad.Tracking.w', calculate_by_hand=args.no_autograd)
        grad = w.grad.data.clone()
        q = bf.synchronize(q_handle) + grad - grad_prev
        grad_prev = grad
        w.grad.data.zero_()

    elif args.computation_mode == "compute_and_no_communicate":
        w.data = - alpha * q
        # calculate local gradient
        logistic_loss_step(
            w, rho, X, y, tensor_name='neighbor.allreduce.Grad.Tracking.w', calculate_by_hand=args.no_autograd)
        grad = w.grad.data.clone()
        q = grad - grad_prev
        grad_prev = grad
        w.grad.data.zero_()

    elif args.computation_mode == "sleep_and_communicate":
        w_handle = bf.neighbor_allreduce_nonblocking(w.data, name='Grad.Tracking.w')
        q_handle = bf.neighbor_allreduce_nonblocking(q, name='Grad.Tracking.q')
        w.data = bf.synchronize(w_handle)
        systemtime.sleep(args.sleep_time)
        q = bf.synchronize(q_handle)

def log(s, nl=True):
    if bf.local_rank() != 0:
        return
    print(s, end='\n' if nl else '', flush=True)

log('Model: %s' % args.task)
log('Data size: %d, Data dims: %d' % (m, n))
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, bf.size()))

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
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
img_secs_sum = bf.allreduce(torch.from_numpy(
    np.array(img_secs)), average=False)
img_sec_mean_all = np.mean(img_secs_sum.numpy())
img_sec_conf_all = 1.96 * np.std(img_secs_sum.numpy())
print('[%d] Img/sec per %s: %.1f +-%.1f' %
      (bf.rank(), device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (bf.size(), device, img_sec_mean_all, img_sec_conf_all))
