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
print("Rank {}: global gradient norm: {}".format(bf.rank(), global_grad_norm))

# the norm of local gradient is expected not be be close to 0
# this is because each rank converges to global solution, not local solution
local_grad_norm = torch.norm(A.T.mm(A.mm(x_opt) - b), p=2)
print("Rank {}: local gradient norm: {}".format(bf.rank(), local_grad_norm))

# Calculate the true solution with exact diffusion:
# Reference: https://arxiv.org/abs/1702.05122
# psi^{k+1} = x^k - alpha * grad(x^k)
# phi^{k+1} = psi^{k+1} + x^k - psi^{k}
# x^{k+1} = neighbor_allreduce(phi^{k+1})
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
# the norm of global gradient is expected to 0 (optimality condition)
global_grad_norm = torch.norm(bf.allreduce(A.T.mm(A.mm(x) - b)), p=2)
print("Rank {}: global gradient norm: {}".format(bf.rank(), global_grad_norm))

# the norm of local gradient is expected not be be close to 0
# this is because each rank converges to global solution, not local solution
local_grad_norm = torch.norm(A.T.mm(A.mm(x) - b), p=2)
print("Rank {}: local gradient norm: {}".format(bf.rank(), local_grad_norm))


if bf.rank() == 0:
    plt.semilogy(mse)
    plt.show()
