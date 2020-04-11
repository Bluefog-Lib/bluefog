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
import copy

bf.init()

# The least squares problem is min_x \sum_i^n \|A_i x - b_i\|^2 
# where each rank i holds A_i and b_i
# we expect each rank will converge to the global solution after the algorithm

# Generate data
# y = A@x + ns where ns is Gaussion noise
m, n = 5, 3
A = torch.randn(m, n)
x = torch.randn(n, 1)
ns = 0.1*torch.randn(m, 1)
b = A.mm(x) + ns

# Calculate the true solution with distributed gradient descent
# x^{k+1} = x^k - alpha * \sum_i A_i.T(A_i x - b_i)
x = torch.zeros(n, 1)
maxite = 1000
alpha = 1e-1
for i in range(maxite):
    grad = A.T.mm(A.mm(x) - b)                  # local gradient
    grad = bf.allreduce(grad, name='gradient')  # global gradient
    x = x - alpha*grad

# evaluate the convergence of distributed least-squares
# the norm of global gradient is expected to 0 (optimality condition)
print("Rank {}: global gradient norm: {}".format(bf.rank(), \
        torch.norm(bf.allreduce(A.T.mm(A.mm(x) - b)))))

# the norm of local gradient is expected not be be close to 0
# this is because each rank converges to global solution, not local solution
print("Rank {}: local gradient norm: {}".format(bf.rank(), \
        torch.norm(A.T.mm(A.mm(x) - b))))

# Calculate the true solution with exact diffusion:
# Reference: https://arxiv.org/abs/1702.05122
# psi^{k+1} = x^k - alpha * grad(x^k)
# phi^{k+1} = psi^{k+1} + x^k - psi^{k}
# x^{k+1} = neighbor_allreduce(phi^{k+1})
x, phi = torch.zeros(n, 1), torch.zeros(n, 1)
psi, psi_prev = torch.zeros(n, 1), torch.zeros(n, 1)
alpha_ed = 1e-1 # step-size for exact diffusion
for i in range(maxite):
    grad = A.T.mm(A.mm(x)-b)    # local gradient
    psi = x - alpha * grad
    phi = psi + x - psi_prev
    x = bf.neighbor_allreduce(phi, name='local variable')
    psi_prev = psi

# evaluate the convergence of exact diffuion least-squares
# the norm of global gradient is expected to 0 (optimality condition)
print("Rank {}: global gradient norm: {}".format(bf.rank(), \
        torch.norm(bf.allreduce(A.T.mm(A.mm(x) - b)))))

# the norm of local gradient is expected not be be close to 0
# this is because each rank converges to global solution, not local solution
print("Rank {}: local gradient norm: {}".format(bf.rank(), \
        torch.norm(A.T.mm(A.mm(x) - b))))
