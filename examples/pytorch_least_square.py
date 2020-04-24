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
import bluefog.torch as bf
from bluefog.common import topology_util
import matplotlib.pyplot as plt

bf.init()

# The least squares problem is min_x \sum_i^n \|A_i x - b_i\|^2
# where each rank i holds A_i and b_i
# we expect each rank will converge to the global solution after the algorithm

# Generate data
# y = A@x + ns where ns is Gaussion noise
torch.random.manual_seed(123417 * bf.rank())
m, n = 20, 5
A = torch.randn(m, n).to(torch.double)
x_o = torch.randn(n, 1).to(torch.double)
ns = 0.1*torch.randn(m, 1).to(torch.double)
b = A.mm(x_o) + ns

# Calculate the solution with distributed gradient descent:
# x^{k+1} = x^k - alpha * \sum_i A_i.T(A_i x - b_i)
# it will be used to verify the solution of exact diffusion.
x_opt = torch.zeros(n, 1).to(torch.double)
maxite = 1000
alpha = 1e-2
for i in range(maxite):
    grad = A.T.mm(A.mm(x_opt) - b)                  # local gradient
    grad = bf.allreduce(grad, name='gradient')  # global gradient
    x_opt = x_opt - alpha*grad

# evaluate the convergence of distributed least-squares
# the norm of global gradient is expected to 0 (optimality condition)
global_grad_norm = torch.norm(bf.allreduce(A.T.mm(A.mm(x_opt) - b)), p=2)
print("[DG] Rank {}: global gradient norm: {}".format(bf.rank(), global_grad_norm))

# the norm of local gradient is expected not be be close to 0
# this is because each rank converges to global solution, not local solution
local_grad_norm = torch.norm(A.T.mm(A.mm(x_opt) - b), p=2)
print("[DG] Rank {}: local gradient norm: {}".format(bf.rank(), local_grad_norm))

# ==================== Exact Diffusion ===========================================
# Calculate the true solution with exact diffusion recursion as follows:
#
# psi^{k+1} = w^k - alpha * grad(w^k)
# phi^{k+1} = psi^{k+1} + w^k - psi^{k}
# w^{k+1} = neighbor_allreduce(phi^{k+1})
#
# References: 
#
# [R1] K. Yuan, B. Ying, X. Zhao, and A. H. Sayed, ``Exact diffusion for distributed
# optimization and learning -- Part I: Algorithm development'', 2018. (Alg. 1)
# link: https://arxiv.org/abs/1702.05122 
#
# [R2] Z. Li, W. Shi and M. Yan, ``A Decentralized Proximal-gradient Method with 
#  Network Independent Step-sizes and Separated Convergence Rates'', 2019
# ================================================================================
x = torch.zeros(n, 1).to(torch.double)
phi, psi, psi_prev = x.clone(), x.clone(), x.clone()
alpha_ed = 1e-2  # step-size for exact diffusion
mse = []
for i in range(maxite):
    grad = A.T.mm(A.mm(x)-b)    # local gradient
    psi = x - alpha * grad
    phi = psi + x - psi_prev
    x = bf.neighbor_allreduce(phi, name='local variable')
    psi_prev = psi
    if bf.rank() == 0:
        mse.append(torch.norm(x - x_opt, p=2))

# evaluate the convergence of exact diffuion least-squares
# the norm of global gradient is expected to be 0 (optimality condition)
global_grad_norm = torch.norm(bf.allreduce(A.T.mm(A.mm(x) - b)), p=2)
print("[ED] Rank {}: global gradient norm: {}".format(bf.rank(), global_grad_norm))

# the norm of local gradient is expected not be be close to 0
# this is because each rank converges to global solution, not local solution
local_grad_norm = torch.norm(A.T.mm(A.mm(x) - b), p=2)
print("[ED] Rank {}: local gradient norm: {}".format(bf.rank(), local_grad_norm))


if bf.rank() == 0:
    plt.semilogy(mse)
    plt.show()

# ======================= gradient tracking =====================================
# Calculate the true solution with gradient tracking (GT for short):
# 
# w^{k+1} = neighbor_allreduce(w^k) - alpha*q^k
# q^{k+1} = neighbor_allreduce(q^k) + grad(w^{k+1}) - grad(w^k)
# where q^0 = grad(w^0)
# 
# References: 
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
x = torch.zeros(n, 1).to(torch.double)
y = A.T.mm(A.mm(x)-b)
grad_prev = y.clone()
alpha_gt = 5e-3  # step-size for GT (should be smaller than exact diffusion)
mse_gt = []
for i in range(maxite):
    x = bf.neighbor_allreduce(x, name='local variable x') - alpha_gt * y
    grad = A.T.mm(A.mm(x)-b)    # local gradient at x^{k+1}
    y = bf.neighbor_allreduce(y, name='local variable y') + grad - grad_prev
    grad_prev = grad
    if bf.rank() == 0:
        mse_gt.append(torch.norm(x - x_opt, p=2))

# evaluate the convergence of gradient tracking for least-squares
# the norm of global gradient is expected to be 0 (optimality condition)
global_grad_norm = torch.norm(bf.allreduce(A.T.mm(A.mm(x) - b)), p=2)
print("[GT] Rank {}: global gradient norm: {}".format(bf.rank(), global_grad_norm))

# the norm of local gradient is expected not be be close to 0
# this is because each rank converges to global solution, not local solution
local_grad_norm = torch.norm(A.T.mm(A.mm(x) - b), p=2)
print("[GT] Rank {}: local gradient norm: {}".format(bf.rank(), local_grad_norm))

if bf.rank() == 0:
    plt.semilogy(mse_gt)
    plt.show()

# ======================= Push-DIGing for directed graph =======================
# Calculate the true solution with Push-DIGing:
#
# Reference: 
#
# [R1] A. Nedic, A. Olshevsky, and W. Shi, ``Achieving geometric convergence 
# for distributed optimization over time-varying graphs'', 2017. (Alg. 2)
# ============================================================================
bf.set_topology(topology_util.PowerTwoRingGraph(bf.size()))
outdegree = len(bf.out_neighbor_ranks())
indegree = len(bf.in_neighbor_ranks())

# u, y, v = w[:n], w[n:2*n], w[2n]
w = torch.zeros(2*n+1, 1).to(torch.double)
x = torch.zeros(n, 1).to(torch.double)
w[n:2*n] = A.T.mm(A.mm(x)-b)
w[-1] = 1.0
grad_prev = w[n:2*n].clone()

bf.win_create(w, name="w_buff", zero_init=True)

alpha_pd = 1e-2  # step-size for Push-DIGing (should be smaller than exact diffusion)
mse_pd = []
maxite = 1000
for i in range(maxite):
    if i%10 == 0:
        bf.barrier()

    w[:n] = w[:n] - alpha_pd*w[n:2*n]
    bf.win_accumulate(
        w, name="w_buff",
        dst_weights={rank: 1.0 / (outdegree + 1)
                     for rank in bf.out_neighbor_ranks()},
        require_mutex=True)
    w.div_(1+outdegree)
    w = bf.win_sync_then_collect(name="w_buff")

    x = w[:n]/w[-1]
    grad = A.T.mm(A.mm(x)-b)
    w[n:2*n] += grad - grad_prev
    grad_prev = grad
    if bf.rank() == 0:
        mse_pd.append(torch.norm(x - x_opt, p=2))

bf.barrier()
w = bf.win_sync_then_collect(name="w_buff")
x = w[:n]/w[-1]

# evaluate the convergence of gradient tracking for least-squares
# the norm of global gradient is expected to be 0 (optimality condition)
global_grad_norm = torch.norm(bf.allreduce(A.T.mm(A.mm(x) - b)), p=2)
print("[PD] Rank {}: global gradient norm: {}".format(bf.rank(), global_grad_norm))

# the norm of local gradient is expected not be be close to 0
# this is because each rank converges to global solution, not local solution
local_grad_norm = torch.norm(A.T.mm(A.mm(x) - b), p=2)
print("[PD] Rank {}: local gradient norm: {}".format(bf.rank(), local_grad_norm))

if bf.rank() == 0:
    plt.semilogy(mse_pd)
    plt.show()