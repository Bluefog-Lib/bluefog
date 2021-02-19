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
import argparse

import torch
import numpy as np
import bluefog.torch as bf
from bluefog.common import topology_util

parser = argparse.ArgumentParser(description='PyTorch Average Consensus',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-size', type=int, default=100000,
                    help='the size of data.')
parser.add_argument('--max-iters', type=int, default=200,
                    help='maximum iterations')
parser.add_argument('--virtual-topology', type=str, default="expo2",
                    help='The underlying virtual topology. Supporting options are ' +
                    '[expo2(Default), ring, mesh, star, InnerOuterExpo2].')
parser.add_argument('--asynchronous-mode', action='store_true', default=False,
                    help='Use one-sided ops to run asynchronous push sum algorithm')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--enable-dynamic-topology', action='store_true',
                    default=False, help=('Enable each iteration to transmit one neighbor ' +
                                         'per iteration dynamically.'))
parser.add_argument(
    "--plot-interactive", action='store_true', help="Use plt.show() to present the plot."
)
parser.add_argument(
    "--save-plot-file", default='average_consensus_plot.png', help="Saving the plot in the file."
)
parser.add_argument('--seed', type=int, default=2020,
                    help='Seed for randomness.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

bf.init()

torch.random.manual_seed(args.seed * bf.rank())
if args.cuda:
    device = bf.local_rank() %  torch.cuda.device_count()
    x = torch.randn(args.data_size, device=device, dtype=torch.double)
else:
    x = torch.randn(args.data_size, dtype=torch.double)

if args.virtual_topology == "expo2":
    pass
elif args.virtual_topology == "expo3":
    bf.set_topology(topology_util.ExponentialGraph(bf.size(), base=3))
elif args.virtual_topology == "expo4":
    bf.set_topology(topology_util.ExponentialGraph(bf.size(), base=4))
elif args.virtual_topology == "ring":
    bf.set_topology(topology_util.RingGraph(bf.size(), connect_style=1))
elif args.virtual_topology == "mesh":
    bf.set_topology(topology_util.RingGraph(
        bf.size(), connect_style=0), is_weighted=True)
elif args.virtual_topology == "star":
    bf.set_topology(topology_util.StarGraph(bf.size()), is_weighted=True)
elif args.virtual_topology == "full":
    bf.set_topology(topology_util.FullyConnectedGraph(bf.size()))
else:
    raise ValueError("Unknown args.virtual_topology, supporting options are " +
                     "[expo2(Default), ring, mesh, star].")

x_bar = bf.allreduce(x, average=True)
mse = [torch.norm(x-x_bar, p=2) / torch.norm(x_bar, p=2)]

if not args.asynchronous_mode:
    self_weight = None
    neighbor_weights = None
    send_neighbors = None

    if args.enable_dynamic_topology:
        if args.virtual_topology == "InnerOuterExpo2":
            dynamic_neighbor_allreduce_gen = topology_util.GetInnerOuterExpo2DynamicSendRecvRanks(
                bf.size(), local_size=bf.local_size(), self_rank=bf.rank())
        else:
            dynamic_neighbor_allreduce_gen = topology_util.GetDynamicOnePeerSendRecvRanks(
                bf.load_topology(), bf.rank())

    for ite in range(args.max_iters):
        if args.enable_dynamic_topology:
            send_neighbors, recv_neighbors = next(dynamic_neighbor_allreduce_gen)
            neighbor_weights = {
                r: 1/(len(recv_neighbors) + 1) for r in recv_neighbors}
            self_weight = 1 / (len(recv_neighbors) + 1)

        x = bf.neighbor_allreduce(x, name='x', self_weight=self_weight,
                                  src_weights=neighbor_weights,
                                  dst_weights=send_neighbors, enable_topo_check=False)
        mse.append(torch.norm(x-x_bar, p=2) / torch.norm(x_bar, p=2))
else:
    outdegree = len(bf.out_neighbor_ranks())
    indegree = len(bf.in_neighbor_ranks())

    if not bf.nccl_built():  # NCCL do not support associated P yet.
        bf.turn_on_win_ops_with_associated_p()
        bf.win_create(x, name="x", zero_init=True)
        for i in range(args.max_iters):
            if args.enable_dynamic_topology:
                num_out_neighbors = len(bf.out_neighbor_ranks())
                sent_neighbor = bf.out_neighbor_ranks()[i % num_out_neighbors]
                dst_weights = {sent_neighbor: 0.5}
                self_weight = 0.5
            else:
                dst_weights = {rank: 1.0 / (outdegree + 1)
                               for rank in bf.out_neighbor_ranks()}
                self_weight = 1/(1+outdegree)

            bf.win_accumulate(x, name="x", self_weight=self_weight,
                              dst_weights=dst_weights, require_mutex=True)
            bf.win_update_then_collect(name="x")
            associated_p = bf.win_associated_p(name="x")
            mse.append(torch.norm(x/associated_p-x_bar, p=2) /
                       torch.norm(x_bar, p=2))

        # Do not forget to sync at last!
        bf.barrier()
        bf.win_update_then_collect(name="x")
        associated_p = bf.win_associated_p(name="x")
        print(f"associated p at {bf.rank()} is {associated_p}")
        bf.turn_off_win_ops_with_associated_p()
        mse.append(torch.norm(x/associated_p - x_bar, p=2) /
                   torch.norm(x_bar, p=2))
        bf.win_free(name="x")
    else:
        p = torch.DoubleTensor([1.0]).to(x.device)
        x_ext = torch.cat([x, p], 0)
        bf.win_create(x_ext, name="x_ext", zero_init=True)
        for i in range(args.max_iters):
            if args.enable_dynamic_topology:
                num_out_neighbors = len(bf.out_neighbor_ranks())
                sent_neighbor = bf.out_neighbor_ranks()[i % num_out_neighbors]
                dst_weights = {sent_neighbor: 0.5}
                self_weight = 0.5
            else:
                dst_weights = {rank: 1.0 / (outdegree + 1)
                               for rank in bf.out_neighbor_ranks()}
                self_weight = 1/(1+outdegree)

            bf.win_accumulate(x_ext, name="x_ext", self_weight=self_weight,
                              dst_weights=dst_weights, require_mutex=True)
            bf.win_update_then_collect(name="x_ext")
            x, associated_p = x_ext[:-1], x_ext[-1]
            mse.append(torch.norm(x/associated_p-x_bar, p=2) /
                       torch.norm(x_bar, p=2))

        # Do not forget to sync at last!
        bf.barrier()
        bf.win_update_then_collect(name="x_ext")
        x, associated_p = x_ext[:-1], x_ext[-1]
        print(f"associated p at {bf.rank()} is {associated_p}")
        mse.append(torch.norm(x/associated_p - x_bar, p=2) /
                   torch.norm(x_bar, p=2))
        bf.win_free(name="x_ext")

    p_push_sum = bf.allreduce(torch.DoubleTensor([associated_p]), average=True)
    if bf.rank() == 0:
        print("Average of p should be the 1 always. Actuall value is ", p_push_sum)


print("MSE at last iteration: ", mse[-1])
if args.plot_interactive and bf.rank() == 0:
    import matplotlib.pyplot as plt
    plt.semilogy(mse)
    plt.savefig(args.save_plot_file)
    plt.show()
    plt.close()
