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

import os
import torch
import matplotlib.pyplot as plt
import argparse
import time
import numpy as np

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
parser.add_argument('--no-autograd', action='store_true', default=False,
                    help='calculate gradient by hand')
parser.add_argument(
    "--non-blocking", action='store_true', help="use non-blocking operations"
)
parser.add_argument(
    "--use-Abar", action='store_true', help="use Abar in exact diffusion algorithm"
)
parser.add_argument('--enable-dynamic-topology', action='store_true',
                    default=False, help=('Enable each iteration to transmit one neighbor ' +
                                         'per iteration dynamically.'))
parser.add_argument(
    "--max-iter", action='store', type=int, default=1000, help="maximum iteration number."
)
parser.add_argument(
    "--total-data-size", action='store', type=int, default=5000, help="total data size."
)
parser.add_argument(
    "--data-dims", action='store', type=int, default=100, help="data dimensions."
)
parser.add_argument(
    "--seed", action='store', type=int, default=123417, help="randomization seed."
)
parser.add_argument(
    "--lr", action='store', type=float, default=1e-1, help="learning rate"
)
parser.add_argument(
    "--tol", action='store', type=float, default=1e-3, help="algorithm will stop when msd reaches tol"
)
parser.add_argument(
    "--method", help="this example supports exact_diffusion, gradient_tracking, and push_diging",
    default='exact_diffusion'
)
parser.add_argument(
    "--task", help="this example supports linear_regression and logistic_regression",
    default='logistic_regression'
)
parser.add_argument(
    "--topology", help="this example supports mesh, star, ring, power_two_ring and symmetric_power_two_ring",
    default='ring'
)
parser.add_argument(
    "--centralized-lr", action='store', type=float, default=1e-1, help="learning rate"
)
parser.add_argument(
    "--centralized-max-iter", action='store', type=int, default=5000, help="maximum iteration number."
)

args = parser.parse_args()
torch.random.manual_seed(args.seed)

def finalize_plot():
    plt.savefig(args.save_plot_file)
    if args.plot_interactive:
        plt.show()
    plt.close()

def generate_data(m, n, task='logistic_regression'):

    if task == 'logistic_regression':
        X = torch.randn(m, n).to(torch.double)
        w_0 = (torch.randn(n, 1)).to(torch.double)
        y = torch.rand(m, 1).to(torch.double) < 1 / (1+torch.exp(X.mm(w_0)))
        y = y.double()
        y = 2*y - 1
    elif task == 'linear_regression':
        X = torch.randn(m, n).to(torch.double)
        x_o = torch.randn(n, 1).to(torch.double)
        ns = 0.1*torch.randn(m, 1).to(torch.double)
        y = X.mm(x_o) + ns
    else:
        raise NotImplementedError(
            'Task not supported. This example only supports' +
            ' linear_regression and logistic_regression')

    return X, y


def _loss_step(X, y, x_, loss='logistic_regression', **kwargs):
    """Calculate gradient via pytorch autograd."""

    calculate_by_hand = kwargs.get('calculate_by_hand', False)

    if loss == 'logistic_regression':
        rho = kwargs.get('rho', 1e-1)

        if not calculate_by_hand:
            loss_ = torch.mean(torch.log(1 + torch.exp(-y*X.mm(x_)))) + \
                        0.5*rho*torch.norm(x_, p=2)

        else:
            loss_ = torch.mean(torch.log(1 + torch.exp(-y*X.mm(x_)))) + \
                        0.5*rho*torch.norm(x_, p=2)
                        
            prob = torch.exp( -y * torch.matmul(X, x_.data))
            alpha = prob / (1+prob)
            x_.grad = rho * x_.data - torch.mean(alpha*y*X, dim = 0).reshape(-1,1)
            return loss_

    elif loss == 'linear_regression':
        loss_ = 0.5*torch.mean(torch.norm(X.mm(x_) - y, p=2))

    else:
        raise NotImplementedError(
            'Task not supported. This example only supports' +
            ' linear_regression and logistic_regression'
        )

    loss_.backward()

    return loss_

def loss_step(X, y, x_, tensor_name, loss='logistic_regression', **kwargs):
    """Calculate gradient via pytorch autograd."""

    if os.getenv('BLUEFOG_TIMELINE'):
        with bf.timeline_context(tensor_name=tensor_name,
                                 activity_name="gradient computation"):
            return _loss_step(X, y, x_, loss=loss, **kwargs)

    else:
        return _loss_step(X, y, x_, loss=loss, **kwargs)

# ================== Distributed gradient descent ================================
# Calculate the solution with distributed gradient descent:
# x^{k+1} = x^k - alpha * allreduce(local_grad)
# it will be used to verify the solution of various decentralized algorithms.
# ================================================================================

def distributed_grad_descent(X, y, loss, maxite=5000, alpha=1e-1, **kwargs):

    if loss == 'logistic_regression':
        rho = kwargs.get('rho', 1e-1)
    elif loss == 'linear_regression':
        rho = 0
    else:
        raise NotImplementedError(
            'Task not supported. This example only supports' +
            ' linear_regression and logistic_regression')

    w_opt = torch.zeros(n, 1, dtype=torch.double, requires_grad=True)

    for _ in range(maxite):
        # calculate gradient via pytorch autograd
        loss_step(X, y, w_opt, tensor_name='allreduce.gradient', \
                            loss=loss, rho=rho, calculate_by_hand=args.no_autograd)
        # global gradient
        grad = bf.allreduce(w_opt.grad.data, name='gradient')

        # distributed gradient descent
        w_opt.data = w_opt.data - alpha*grad
        w_opt.grad.data.zero_()

    loss_step(X, y, w_opt, tensor_name='allreduce.gradient', \
                        loss=loss, rho=rho, calculate_by_hand=args.no_autograd)
    grad = bf.allreduce(w_opt.grad.data, name='gradient')  # global gradient

    # evaluate the convergence of distributed logistic regression
    # the norm of global gradient is expected to 0 (optimality condition)
    global_grad_norm = torch.norm(grad, p=2)
    print("[DG] Rank {}: global gradient norm: {}".format(
        bf.rank(), global_grad_norm))

    # the norm of local gradient is expected not be be close to 0
    # this is because each rank converges to global solution, not local solution
    local_grad_norm = torch.norm(w_opt.grad.data, p=2)
    print("[DG] Rank {}: local gradient norm: {}".format(
        bf.rank(), local_grad_norm))

    return w_opt

# ==================== Diffusion ================================================
# Calculate the solution with diffusion:
# x^{k+1} = allreduce(x^k - alpha * local_grad)
#
# Reference:
# 
# [R1] A. H. Sayed, ``Adaptive networks'', Proceedings of the IEEE, 2014
#
# [R2] A. H. Sayed, ``Adaptation, learning, and optimization over networks'', 2014
# ================================================================================
def diffusion(X, y, w_opt, loss, maxite=2000, alpha=1e-1, **kwargs):
    
    if loss == 'logistic_regression':
        rho = kwargs.get('rho', 1e-1)
    elif loss == 'linear_regression':
        rho = 0
    else:
        raise NotImplementedError(
            'Task not supported. This example only supports' +
            ' linear_regression and logistic_regression')

    topology = bf.load_topology()
    self_weight, neighbor_weights = topology_util.GetWeights(topology, bf.rank())

    w = torch.zeros(n, 1, dtype=torch.double, requires_grad=True)
    phi = w.clone()
    mse = []

    for i in range(maxite):
        # calculate loccal gradient via pytorch autograd
        loss_step(X, y, w, \
            tensor_name='neighbor.allreduce.local_variable', loss=loss, rho=rho, calculate_by_hand=args.no_autograd)

        # diffusion
        phi = w - alpha * w.grad.data
        w.data = bf.neighbor_allreduce(
            phi, self_weight, neighbor_weights, name='local variable')
        w.grad.data.zero_()

        # record convergence
        if bf.rank() == 0:
            mse.append(torch.norm(w.data - w_opt.data, p=2))

    return w, mse

# ==================== Exact Diffusion ===========================================
# Calculate the true solution with exact diffusion recursion as follows:
#
# psi^{k+1} = w^k - alpha * grad(w^k)
# phi^{k+1} = psi^{k+1} + w^k - psi^{k}
# w^{k+1} = neighbor_allreduce(phi^{k+1})
#
# Reference:
#
# [R1] K. Yuan, B. Ying, X. Zhao, and A. H. Sayed, ``Exact diffusion for distributed
# optimization and learning -- Part I: Algorithm development'', 2018. (Alg. 1)
# link: https://arxiv.org/abs/1702.05122
#
# [R2] Z. Li, W. Shi and M. Yan, ``A Decentralized Proximal-gradient Method with
#  Network Independent Step-sizes and Separated Convergence Rates'', 2019
# ================================================================================
def exact_diffusion(X, y, w_opt, loss, maxite=2000, alpha=1e-1, use_Abar=True, **kwargs):

    time_start = time.time()
    tol_reached = False
    tol = kwargs.get('tol', 1e-6)

    if loss == 'logistic_regression':
        rho = kwargs.get('rho', 1e-1)
    elif loss == 'linear_regression':
        rho = 0
    else:
        raise NotImplementedError(
            'Task not supported. This example only supports' +
            ' linear_regression and logistic_regression'
        )

    topology = bf.load_topology()
    # self_weight, neighbor_weights = topology_util.GetWeights(
    #     topology, bf.rank())
    self_weight, neighbor_weights = topology_util.GetRecvWeights(
        topology, bf.rank())

    if bf.rank() == 0:
        print('self weights with A: {}\n'.format(self_weight))
        print('neighbor weights with A:\n')
        for k, v in neighbor_weights.items():
            print(k, v)

    w = torch.zeros(n, 1, dtype=torch.double, requires_grad=True)
    phi, psi, psi_prev = w.clone(), w.clone(), w.clone()
    mse = []

    # construct A_bar
    if use_Abar:
        self_weight = (self_weight+1)/2
        for k, v in neighbor_weights.items():
            neighbor_weights[k] = v/2

    for i in range(maxite):
        # calculate loccal gradient via pytorch autograd
        loss_step(X, y, w, tensor_name='neighbor.allreduce.local_variable', \
                            loss=loss, rho=rho,
                            calculate_by_hand=args.no_autograd)

        if args.enable_dynamic_topology:
            self_weight, neighbor_weights, send_neighbors, enable_topo_check = dynamic_topology_update() 

            if bf.rank() == 0 and i < 4:
                print('self weights with A: {}\n'.format(self_weight))
                print('neighbor weights with A:\n')
                for k, v in neighbor_weights.items():
                    print(k, v)

            # construct A_bar
            # do we need to change send_neighbors?
            if use_Abar:
                self_weight = (self_weight+1)/2
                for k, v in neighbor_weights.items():
                    neighbor_weights[k] = v/2    

            # exact diffusion
            psi = w - alpha * w.grad.data
            phi = psi + w.data - psi_prev
            w.data = bf.neighbor_allreduce(phi, self_weight, neighbor_weights, 
                      send_neighbors, enable_topo_check, name='local variable')
            psi_prev = psi.clone()
            w.grad.data.zero_()  

        else:  
            # exact diffusion
            psi = w - alpha * w.grad.data
            phi = psi + w.data - psi_prev
            w.data = bf.neighbor_allreduce(phi, self_weight, neighbor_weights, name='local variable')
            psi_prev = psi.clone()
            w.grad.data.zero_()

        # record convergence
        if bf.rank() == 0:
            mse.append(torch.norm(w.data - w_opt.data, p=2))

    # record convergence
    if bf.rank() == 0:
        mse_at_current_point = torch.norm(w.data - w_opt.data, p=2)/torch.norm(w_opt.data, p=2)
        mse.append(mse_at_current_point)
    if torch.norm(w.data - w_opt.data, p=2)/torch.norm(w_opt.data, p=2) < tol and not tol_reached:
        time_end = time.time()
        print('rank:{}, tol:{}, bluefog iterations:{}, time collapsed:{}'.format(bf.rank(),
                torch.norm(w.data - w_opt.data, p=2)/torch.norm(w_opt.data, p=2), i, time_end - time_start))
        tol_reached = True

    i += 1

    return w, mse

# ======================= gradient tracking =====================================
# Calculate the true solution with gradient tracking (GT for short):
#
# w^{k+1} = neighbor_allreduce(w^k) - alpha*q^k
# q^{k+1} = neighbor_allreduce(q^k) + grad(w^{k+1}) - grad(w^k)
# where q^0 = grad(w^0)
#
# Reference:
# [R1] A. Nedic, A. Olshevsky, and W. Shi, ``Achieving geometric convergence
# for distributed optimization over time-varying graphs'', 2017. (Alg. 1)
#
# [R2] G. Qu and N. Li, ``Harnessing smoothness to accelerate distributed
# optimization'', 2018
#
# [R3] J. Xu et.al., ``Augmented distributed gradient methods for multi-agent
# optimization under uncoordinated constant stepsizes'', 2015
#
# [R4] P. Di Lorenzo and G. Scutari, ``Next: In-network nonconvex optimization'',
# 2016
# ================================================================================
def gradient_tracking(X, y, w_opt, loss, maxite=2000, alpha=1e-1, **kwargs):
    time_start = time.time()
    tol_reached = False

    if loss == 'logistic_regression':
        rho = kwargs.get('rho', 1e-1)
    elif loss == 'linear_regression':
        rho = 0
    else:
        raise NotImplementedError(
            'Task not supported. This example only supports' +
            ' linear_regression and logistic_regression')

    non_blocking = kwargs.get('non_blocking', False)
    tol = kwargs.get('tol', 1e-6)

    w = torch.zeros(n, 1, dtype=torch.double, requires_grad=True)
    loss_step(X, y, w, tensor_name='neighbor.allreduce.Grad.Tracking.w', \
                        loss=loss, rho=rho,
                        calculate_by_hand=args.no_autograd)
    q = w.grad.data.clone()  # q^0 = grad(w^0)
    w.grad.data.zero_()

    grad_prev = q.clone()
    mse = []

    # for i in range(maxite):
    i = 0

    try:
        while i < maxite:

            # Algorithm:
            # w^{k+1} = neighbor_allreduce(w^k) - alpha*q^k
            # q^{k+1} = neighbor_allreduce(q^k) + grad(w^{k+1}) - grad(w^k)

            if not non_blocking:
                # Notice the communication of neighbor_allreduce can overlap with gradient computation.

                w.data = bf.neighbor_allreduce(w.data, name='Grad.Tracking.w')
                w.data = w.data -  alpha * q
                q = bf.neighbor_allreduce(q, name='Grad.Tracking.q')

                # calculate local gradient
                loss_step(X, y, w, tensor_name='neighbor.allreduce.Grad.Tracking.w', \
                                loss=loss, rho=rho,
                                calculate_by_hand=args.no_autograd)
                grad = w.grad.data.clone()
                # q = bf.synchronize(q_handle) + grad - grad_prev
                q = q + grad - grad_prev
                grad_prev = grad
                w.grad.data.zero_()
            else:

                if args.enable_dynamic_topology:
                    self_weight, neighbor_weights, send_neighbors, enable_topo_check = dynamic_topology_update()                    

                    w_handle = bf.neighbor_allreduce_nonblocking(w.data, self_weight, neighbor_weights,
                                send_neighbors, enable_topo_check, name='Grad.Tracking.w')
                    w.data = - alpha * q + bf.synchronize(w_handle)
                    q_handle = bf.neighbor_allreduce_nonblocking(q, self_weight, neighbor_weights,
                                send_neighbors, enable_topo_check, name='Grad.Tracking.q')

                else:
                    w_handle = bf.neighbor_allreduce_nonblocking(w.data, name='Grad.Tracking.w')
                    w.data = - alpha * q + bf.synchronize(w_handle)
                    q_handle = bf.neighbor_allreduce_nonblocking(q, name='Grad.Tracking.q')

                # q = bf.synchronize(q_handle)
                # calculate local gradient
                loss_step(X, y, w, tensor_name='neighbor.allreduce.Grad.Tracking.w', \
                                loss=loss, rho=rho,
                                calculate_by_hand=args.no_autograd)
                grad = w.grad.data.clone()
                q = grad - grad_prev + bf.synchronize(q_handle)
                grad_prev = grad
                w.grad.data.zero_()

            # record convergence
            if bf.rank() == 0:
                mse_at_current_point = torch.norm(w.data - w_opt.data, p=2)/torch.norm(w_opt.data, p=2)
                mse.append(mse_at_current_point)
            if torch.norm(w.data - w_opt.data, p=2)/torch.norm(w_opt.data, p=2) < tol and not tol_reached:
                time_end = time.time()
                print('rank:{}, tol:{}, bluefog iterations:{}, time collapsed:{}'.format(bf.rank(),
                        torch.norm(w.data - w_opt.data, p=2)/torch.norm(w_opt.data, p=2), i, time_end - time_start),
                        flush=True)
                tol_reached = True

            i += 1

    except RuntimeError:
        print("rank {}: Desiered tolerance is reached.".format(bf.rank()))

    return w, mse

# ======================= Push-DIGing for directed graph =======================
# Calculate the true solution with Push-DIGing:
# 
# u^{k+1} = directed_neighbor_allreduce(u^k - alpha y^k)
# v^{k+1} = directed_neighbor_allreduce(v^k)
# x^{k+1} = (v^{k+1})^{-1}*u^{k+1}
# y^{k+1} = directed_neighbor_allreduce(y^k) + grad(x^{k+1}) - grad(x^k)
#
# Reference:
#
# [R1] A. Nedic, A. Olshevsky, and W. Shi, ``Achieving geometric convergence
# for distributed optimization over time-varying graphs'', 2017. (Alg. 2)
# ============================================================================
def push_diging(X, y, w_opt, loss, maxite=2000, alpha=1e-1, **kwargs):

    if loss == 'logistic_regression':
        rho = kwargs.get('rho', 1e-1)
    elif loss == 'linear_regression':
        rho = 0
    else:
        raise NotImplementedError(
            'Task not supported. This example only supports' +
            ' linear_regression and logistic_regression')

    outdegree = len(bf.out_neighbor_ranks())
    indegree = len(bf.in_neighbor_ranks())

    # We let w = col{u, y, v}, i.e., u, y, v = w[:n], w[n:2*n], w[2n]
    # Insteady of three directed_neighbor_allreduce operations for u, y,
    # and v respectively, we exploit one directed_neighbor_allreduce for
    # the combo vector w. This guarantees u, y, and v to be transmitted
    # simultanesly and avoids the mismatch between them. Experiments 
    # show directed_neighbor_allreduce(w) is crutial for convergence of
    # push_diging.
    w = torch.zeros(2*n+1, 1).to(torch.double)
    x = torch.zeros(n, 1, dtype=torch.double, requires_grad=True)
    loss_step(X, y, x, tensor_name='w_buff',loss=loss, rho=rho, calculate_by_hand=args.no_autograd)

    grad = x.grad.data.clone()
    w[n:2*n] = grad
    x.grad.data.zero_()

    w[-1] = 1.0
    grad_prev = w[n:2*n].clone()

    bf.win_create(w, name="w_buff", zero_init=True)

    mse = []
    for _ in range(maxite):
        bf.barrier()

        w[:n] = w[:n] - alpha*w[n:2*n]
        bf.win_accumulate(
            w, name="w_buff",
            dst_weights={rank: 1.0 / (outdegree*2)
                         for rank in bf.out_neighbor_ranks()},
            require_mutex=True)
        w.div_(2)
        bf.barrier()

        w = bf.win_update_then_collect(name="w_buff")

        x.data = w[:n]/w[-1]
        loss_step(X, y, x, tensor_name='w_buff',loss=loss, rho=rho, calculate_by_hand=args.no_autograd)
        grad = x.grad.data.clone()
        x.grad.data.zero_()

        w[n:2*n] += grad - grad_prev
        grad_prev = grad
        if bf.rank() == 0:
            mse.append(torch.norm(x.data - w_opt, p=2))

    bf.barrier()
    w = bf.win_update_then_collect(name="w_buff")
    x.data = w[:n]/w[-1]

    return x, mse


# ======================= Code starts here =======================
bf.init()

if args.topology == 'mesh':
    bf.set_topology(topology_util.MeshGrid2DGraph(bf.size()), is_weighted=True)
elif args.topology == 'power_two_ring':
    bf.set_topology(topology_util.PowerTwoRingGraph(bf.size()))
elif args.topology == 'symmetric_power_two_ring':
    bf.set_topology(topology_util.SymmetricPowerGraph(bf.size()))
elif args.topology == 'star':
    bf.set_topology(topology_util.StarGraph(bf.size()), is_weighted=True)
elif args.topology == 'ring':
    bf.set_topology(topology_util.RingGraph(bf.size()))
else:
    raise NotImplementedError(
        'Topology not supported. This example only supports' +
        ' mesh, star, ring and power_two_ring'
    )

if args.enable_dynamic_topology:
    dynamic_neighbor_allreduce_gen = topology_util.GetDynamicSendRecvRanks(
        bf.load_topology(), bf.rank())

def dynamic_topology_update():
    send_neighbor, recv_neighbors = next(dynamic_neighbor_allreduce_gen)
    send_neighbors = [send_neighbor]
    neighbor_weights = {
        r: 1/(len(recv_neighbors) + 1) for r in recv_neighbors}
    self_weight = 1 / (len(recv_neighbors) + 1)
    enable_topo_check = False
    return self_weight, neighbor_weights, send_neighbors, enable_topo_check

# Generate data for logistic regression (synthesized data)
m_total, n = args.total_data_size, args.data_dims
assert m_total%bf.size() == 0, "Total data size must be multiple of bf.size()"
m = int(args.total_data_size/bf.size())
rho = 1e-2
X_total, y_total = generate_data(m_total, n, task=args.task)
X = X_total[bf.rank()*m:(bf.rank()+1)*m]
y = y_total[bf.rank()*m:(bf.rank()+1)*m]
X_np = X.numpy()
print("rank {}: data size {}, dims {}, Lipschitz constant {}".format(bf.rank(), m, n, np.linalg.norm(X_np,ord=2)))

# calculate the global solution w_opt via distributed gradient descent
w_opt = distributed_grad_descent(X, y, loss=args.task,
                                    maxite=args.centralized_max_iter,
                                    alpha=args.centralized_lr, rho=rho)

# solve the logistic regression with indicated decentralized algorithms
print('max iteration is:', args.max_iter)
if args.method == 'diffusion':
    w, mse = diffusion(X, y, w_opt, loss=args.task,
                        maxite=args.max_iter, alpha=args.lr, rho=rho)
elif args.method == 'exact_diffusion':
    w, mse = exact_diffusion(X, y, w_opt, loss=args.task, \
                                maxite=args.max_iter, alpha=args.lr, \
                                use_Abar=args.use_Abar, rho=rho, tol=args.tol)
elif args.method == 'gradient_tracking':
    w, mse = gradient_tracking(X, y, w_opt, loss=args.task, \
                maxite=args.max_iter, alpha=args.lr, rho=rho, non_blocking=args.non_blocking, tol=args.tol)
elif args.method == 'push_diging':
    w, mse = push_diging(X, y, w_opt, loss=args.task, \
                maxite=args.max_iter, alpha=args.lr, rho=rho)
else:
    raise NotImplementedError(
        'Algorithm not support. This example only supports' +
        ' exact_diffusion, gradient_tracking, and push_diging'
    )

# plot and print result
if bf.rank() == 0:
    plt.semilogy(mse)
    finalize_plot()

# calculate local and global gradient
loss_step(X, y, w, tensor_name='w_buff', loss=args.task, rho=rho, calculate_by_hand=args.no_autograd)
grad = bf.allreduce(w.grad.data, name='gradient')  # global gradient

# evaluate the convergence of gradient tracking for logistic regression
# the norm of global gradient is expected to be 0 (optimality condition)
global_grad_norm = torch.norm(grad, p=2)
print("[{}] Rank {}: global gradient norm: {}".format(
    args.method, bf.rank(), global_grad_norm))

# the norm of local gradient is expected not to be close to 0
# this is because each rank converges to global solution, not local solution
local_grad_norm = torch.norm(w.grad.data, p=2)
print("[{}] Rank {}: local gradient norm: {}".format(
    args.method, bf.rank(), local_grad_norm))
