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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import bluefog.torch as bf

TEST_ON_GPU = torch.cuda.is_available()

# A linear model for testing
class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# A Simple dataset for testing.
class SimpleDataset:
    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    def __len__(self):
        return len(self._y)
    
    def __getitem__(self, idx):
        return (torch.tensor(self._x[idx],dtype=torch.float32),
                torch.tensor(self._y[idx],dtype=torch.float32))

# A ProblemBuilder for the linear problem with a specified input and output dimension.
# The matrix A are randomly generated now.
#   y = AX + e, e ~ N(0, noise_level^2)
class LinearProblemBuilder:
    def __init__(self, input_dim = 16, output_dim = 3, noise_level = 1e-5):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._noise_level = noise_level
        self._matrix_gen_seed = 0
        self._generate_matrices()

    def _generate_matrices(self):
        state = np.random.get_state()
        np.random.seed(self._matrix_gen_seed)
        self._A = np.random.randn(self._input_dim, self._output_dim)
        #self._A = np.ones((self._input_dim, self._output_dim))
        np.random.set_state(state)

    @property
    def input_dim(self):
        return self._input_dim
    
    @input_dim.setter
    def input_dim(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Input dimension should be an integer larger than 0.")
        self._input_dim = value
        self._generate_matrices()

    @property
    def output_dim(self):
        return self._output_dim
    
    @output_dim.setter
    def output_dim(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Output dimension should be an integer larger than 0.")
        self._output_dim = value
        self._generate_matrices()

    @property
    def noise_level(self):
        return self._noise_level

    @noise_level.setter
    def noise_level(self, value):
        if value < 0:
            raise ValueError("Noise level should be larger than or equal to 0.")
        self._noise_level = value

    def get_dataset(self, num_sample):
        x = np.random.randn(num_sample, self.input_dim)
        e = np.random.randn(num_sample, self.output_dim) * self.noise_level
        y = np.matmul(x, self._A) + e
        return SimpleDataset(x, y)

def problem_setup():
    bf.init()
    num_epochs = 50
    batch_size = 128
    num_train_per_node = 1000
    num_test_per_node = 100
    lr = 0.05

    # Setup Problem
    problem_builder = LinearProblemBuilder()
    train_dataset = problem_builder.get_dataset(num_train_per_node)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = problem_builder.get_dataset(num_test_per_node)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # Setup Model
    model = LinearNet(problem_builder.input_dim, problem_builder.output_dim)
    # Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr*bf.size())
    bf.broadcast_parameters(model.state_dict(), root_rank=0)
    bf.broadcast_optimizer_state(optimizer, root_rank=0)
    return problem_builder, train_dataloader, test_dataloader, model, optimizer, num_epochs

# Standard training process
def standard_train(model, optimizer, dataloader, isCUDA):
    mseloss = nn.MSELoss()
    model.train()
    for data, target in dataloader:
        if isCUDA:
            data, target = data.cuda(), target.cuda()
        y = model(data)
        loss = mseloss(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluation(model, dataloader, isCUDA):
    mseloss = nn.MSELoss()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            if isCUDA:
                data, target = data.cuda(), target.cuda()
            y = model(data)
            loss = mseloss(y, target)
            total_loss += loss * len(target)
        total_loss /= len(dataloader.dataset)
    avg_total_loss = bf.allreduce(total_loss)
    return avg_total_loss.item()

test_scenarios = []
test_scenarios.append(pytest.param("CPU", bf.CommunicationType.empty, False, 2,
                                   id="AWC Empty on CPU"))
test_scenarios.append(pytest.param("CPU", bf.CommunicationType.empty, True, 2,
                                   id="ATC Empty on CPU"))
test_scenarios.append(pytest.param("CPU", bf.CommunicationType.allreduce, False, 1.5,
                                   id="AWC Allreduce on CPU"))
test_scenarios.append(pytest.param("CPU", bf.CommunicationType.allreduce, True, 1.5,
                                   id="ATC Allreduce on CPU"))
test_scenarios.append(pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, False, 1.5,
                                   id="AWC Neighbor Allreduce on CPU"))
test_scenarios.append(pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, True, 1.5,
                                   id="ATC Neighbor Allreduce on CPU"))
# test_scenarios.append(pytest.param("CPU", "win.put", False,
                                #    id="Window put on CPU"))
test_scenarios.append(pytest.param("CPU", "gradient.allreduce", False, 1.5,
                                   id="Gradient Allreduce on CPU"))
if TEST_ON_GPU:
    test_scenarios.append(pytest.param("GPU", bf.CommunicationType.empty, False, 2,
                                       id="AWC Empty on GPU"))
    test_scenarios.append(pytest.param("GPU", bf.CommunicationType.empty, True, 2,
                                       id="ATC Empty on GPU"))
    test_scenarios.append(pytest.param("GPU", bf.CommunicationType.allreduce, False, 1.5,
                                       id="AWC Allreduce on GPU"))
    test_scenarios.append(pytest.param("GPU", bf.CommunicationType.allreduce, True, 1.5,
                                       id="ATC Allreduce on GPU"))
    test_scenarios.append(pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, False, 1.5,
                                       id="AWC Neighbor Allreduce on GPU"))
    test_scenarios.append(pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, True, 1.5,
                                       id="ATC Neighbor Allreduce on GPU"))
    # test_scenarios.append(pytest.param("GPU", "win.put", False,
                                    #    id="Window put on GPU"))
    test_scenarios.append(pytest.param("GPU", "gradient.allreduce", False, 1.5,
                                       id="Gradient Allreduce on GPU"))

@pytest.mark.parametrize("device,communication_type,atc_style,allowed_error", test_scenarios)
def test_optimizer(device, communication_type, atc_style, allowed_error):
    problem_builder, train_dataloader, test_dataloader, model, optimizer, num_epochs = \
        problem_setup()

    isCUDA = device=="GPU"
    if isCUDA:
        # Bluefog: pin GPU to local rank.
        device_id = (bf.local_rank() if bf.nccl_built() else
                     bf.local_rank() % torch.cuda.device_count())
        torch.cuda.set_device(device_id)
        model.cuda()

    if isinstance(communication_type, bf.CommunicationType):
        base_dist_optimizer = (bf.DistributedAdaptThenCombineOptimizer if atc_style else
                               bf.DistributedAdaptWithCombineOptimizer)
        optimizer = base_dist_optimizer(optimizer, model=model,
                                        communication_type=communication_type)
    elif communication_type == "win.put":
        optimizer = bf.DistributedWinPutOptimizer(optimizer, model=model)
    elif communication_type == "gradient.allreduce":
        optimizer = bf.DistributedGradientAllreduceOptimizer(optimizer, model=model)
    # TODO: Dynamic topology and J

    # Train and test
    train_mse = []
    test_mse = []
    for epoch in range(num_epochs):
        standard_train(model, optimizer, train_dataloader, isCUDA)
        train_mse.append(evaluation(model, train_dataloader, isCUDA))
        test_mse.append(evaluation(model, test_dataloader, isCUDA))
    train_mse = np.array(train_mse)
    test_mse = np.array(test_mse)

    # Check if the MSEs in the last three epochs are small enough
    assert (
        train_mse[-3:].max() < allowed_error*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."
    assert (
        test_mse[-3:].max() < allowed_error*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."