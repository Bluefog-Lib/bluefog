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

# Note this end-to-end only covers the BlueFog optimizers for a single machine,
# all the hierarchical cases are not fully under test yet.

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
from bluefog.common import topology_util

TEST_ON_GPU = torch.cuda.is_available()

# A linear model for testing
class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self._num_parameters = input_dim*output_dim

    def forward(self, x):
        return self.fc(x)

    @property
    def num_parameters(self):
        return self._num_parameters

# A deep linear model for testing
class DuplicatedLinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuplicatedLinearNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self._num_parameters = input_dim*output_dim+output_dim**2

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc2(x)
        return x

    @property
    def num_parameters(self):
        return self._num_parameters


class HierarchicalLinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HierarchicalLinearNet, self).__init__()
        self.fc1 = LinearNet(input_dim, output_dim)
        self.fc2 = LinearNet(output_dim, output_dim)
        self._num_parameters = self.fc1.num_parameters+self.fc2.num_parameters

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    @property
    def num_parameters(self):
        return self._num_parameters

# A Simple dataset for testing.
class SimpleDataset:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        return (torch.tensor(self._x[idx], dtype=torch.float32),
                torch.tensor(self._y[idx], dtype=torch.float32))

# A ProblemBuilder for the linear problem with a specified input and output dimension.
# The matrix A are randomly generated now.
#   Y = XA + E, E ~ N(0, noise_level^2)
class LinearProblemBuilder:
    def __init__(self, input_dim=16, output_dim=3, noise_level=1e-5):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._noise_level = noise_level
        self._matrix_gen_seed = 0
        self._generate_matrices()

    def _generate_matrices(self):
        state = np.random.get_state()
        np.random.seed(self._matrix_gen_seed)
        self._A = np.random.randn(self._input_dim, self._output_dim)
        np.random.set_state(state)

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                "Input dimension should be an integer larger than 0.")
        self._input_dim = value
        self._generate_matrices()

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                "Output dimension should be an integer larger than 0.")
        self._output_dim = value
        self._generate_matrices()

    @property
    def noise_level(self):
        return self._noise_level

    @noise_level.setter
    def noise_level(self, value):
        if value < 0:
            raise ValueError(
                "Noise level should be larger than or equal to 0.")
        self._noise_level = value

    def get_dataset(self, num_sample):
        X = np.random.randn(num_sample, self.input_dim)
        E = np.random.randn(num_sample, self.output_dim) * self.noise_level
        Y = np.matmul(X, self._A) + E
        return SimpleDataset(X, Y)

# Prepare the problem  to be solved
def problem_setup(net=LinearNet):
    bf.init()
    num_epochs = 50
    batch_size = 128
    num_train_per_node = 1024
    num_test_per_node = 128
    lr = 0.01

    # Setup Problem
    problem_builder = LinearProblemBuilder()
    train_dataset = problem_builder.get_dataset(num_train_per_node)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = problem_builder.get_dataset(num_test_per_node)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # Setup Model
    model = net(problem_builder.input_dim, problem_builder.output_dim)
    assert (
        num_train_per_node*bf.size() >= model.num_parameters
    ), "The number of samples is too small making it an underdetermined system."
    # Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr*bf.size())
    bf.broadcast_parameters(model.state_dict(), root_rank=0)
    bf.broadcast_optimizer_state(optimizer, root_rank=0)
    return problem_builder, train_dataloader, test_dataloader, model, optimizer, num_epochs


def pin_model_to_device(device, model):
    isCUDA = device == "GPU"
    if isCUDA:
        # Bluefog: pin GPU to local rank.
        device_id = (bf.local_rank() if bf.nccl_built() else
                     bf.local_rank() % torch.cuda.device_count())
        torch.cuda.set_device(device_id)
        model.cuda()
    return isCUDA

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

# Dynamic topology training process for win_put
def dynamic_win_put_train(model, optimizer, dataloader, isCUDA, epoch):
    mseloss = nn.MSELoss()
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        if epoch < 3:
            return
        num_out_neighbors = len(bf.out_neighbor_ranks())
        sent_neighbor = bf.out_neighbor_ranks()[batch_idx % num_out_neighbors]
        optimizer.dst_weights = {sent_neighbor: 1.0}
        if isCUDA:
            data, target = data.cuda(), target.cuda()
        y = model(data)
        loss = mseloss(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Dynamic topology training process for neighbor_allreduce
def dynamic_neighbor_allreduce_train(model, optimizer, dataloader, isCUDA, dynamic_topo_gen):
    mseloss = nn.MSELoss()
    model.train()
    for data, target in dataloader:
        send_neighbors, recv_neighbors = next(dynamic_topo_gen)
        optimizer.dst_weights = send_neighbors
        optimizer.src_weights = {
            r: 1/(len(recv_neighbors) + 1) for r in recv_neighbors}
        optimizer.self_weight = 1 / (len(recv_neighbors) + 1)

        if isCUDA:
            data, target = data.cuda(), target.cuda()
        y = model(data)
        loss = mseloss(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Training process with mini_batch, expecting optimizer with a preset num_step_per_communication
# TODO(ybc,hanbinhu): left-over data if batch_size cannot be divided by num_step_per_communication.
def local_aggregation_train(model, optimizer, dataloader, isCUDA, mini_batch_size):
    mseloss = nn.MSELoss()
    model.train()
    for data, target in dataloader:
        if isCUDA:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        for i in range(0, len(data), mini_batch_size):
            data_batch = data[i: i + mini_batch_size]
            target_batch = target[i: i + mini_batch_size]
            y = model(data_batch)
            loss = mseloss(y, target_batch)
            loss.div_(len(data) / mini_batch_size)
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


static_topo_scenarios = []
static_topo_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.empty, {"ATC": False, "error_threshold": 2},
                 id="AWC Empty on CPU"))
static_topo_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.empty, {"ATC": True, "error_threshold": 2},
                 id="ATC Empty on CPU"))
static_topo_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.allreduce, {"ATC": False},
                 id="AWC Allreduce on CPU"))
static_topo_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.allreduce, {"ATC": True},
                 id="ATC Allreduce on CPU"))
static_topo_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"ATC": False},
                 id="AWC Neighbor Allreduce on CPU"))
static_topo_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"ATC": True},
                 id="ATC Neighbor Allreduce on CPU"))
static_topo_scenarios.append(
    pytest.param("CPU", "gradient.allreduce", {}, id="Gradient Allreduce on CPU"))
static_topo_scenarios.append(
    pytest.param("CPU", "win.put", {'window_prefix': 'CPU'}, id="Window put on CPU"))
if TEST_ON_GPU:
    static_topo_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.empty, {"ATC": False, "error_threshold": 2},
                     id="AWC Empty on GPU"))
    static_topo_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.empty, {"ATC": True, "error_threshold": 2},
                     id="ATC Empty on GPU"))
    static_topo_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.allreduce, {"ATC": False},
                     id="AWC Allreduce on GPU"))
    static_topo_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.allreduce, {"ATC": True},
                     id="ATC Allreduce on GPU"))
    static_topo_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, {"ATC": False},
                     id="AWC Neighbor Allreduce on GPU"))
    static_topo_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, {"ATC": True},
                     id="ATC Neighbor Allreduce on GPU"))
    static_topo_scenarios.append(
        pytest.param("GPU", "gradient.allreduce", {}, id="Gradient Allreduce on GPU"))
    static_topo_scenarios.append(
        pytest.param("GPU", "win.put", {'window_prefix': 'GPU'}, id="Window put on GPU"))

# device can be set to "GPU" or "CPU".
# communication_type can be selected from bf.CommunicationType, "gradient.allreduce" or "win.put".
# kwargs is some optional parameters related to certain communication types.
@pytest.mark.parametrize("device,communication_type,kwargs", static_topo_scenarios)
def test_standard_optimizer(device, communication_type, kwargs):
    atc_style = kwargs.get("ATC", False)
    error_threshold = kwargs.get("error_threshold", 1.5)
    window_prefix = kwargs.get("window_prefix", None)

    problem_builder, train_dataloader, test_dataloader, model, optimizer, num_epochs = \
        problem_setup()

    isCUDA = pin_model_to_device(device, model)

    if isinstance(communication_type, bf.CommunicationType):
        base_dist_optimizer = (bf.DistributedAdaptThenCombineOptimizer if atc_style else
                               bf.DistributedAdaptWithCombineOptimizer)
        optimizer = base_dist_optimizer(optimizer, model=model,
                                        communication_type=communication_type)
    elif communication_type == "win.put":
        optimizer = bf.DistributedWinPutOptimizer(optimizer, model=model,
                                                  window_prefix=window_prefix)
    elif communication_type == "gradient.allreduce":
        optimizer = bf.DistributedGradientAllreduceOptimizer(
            optimizer, model=model)
    else:
        raise ValueError("Communication_type under test is not expected.")

    # Train and test
    train_mse = []
    test_mse = []
    for _ in range(num_epochs):
        standard_train(model, optimizer, train_dataloader, isCUDA)
        train_mse.append(evaluation(model, train_dataloader, isCUDA))
        test_mse.append(evaluation(model, test_dataloader, isCUDA))
    train_mse = np.array(train_mse)
    test_mse = np.array(test_mse)

    # Check if the MSEs in the last three epochs are small enough
    assert (
        train_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."
    assert (
        test_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."

    if communication_type == "win.put":
        optimizer.unregister_window()

hierarchical_model_scenarios = []
hierarchical_model_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"ATC": False},
                 id="AWC Neighbor Allreduce on CPU",
                 marks=pytest.mark.skip(reason="AWC doesn't converge for hierarchical model.")))
hierarchical_model_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"ATC": True},
                 id="ATC Neighbor Allreduce on CPU",
                 marks=pytest.mark.skip(reason="ATC doesn't converge for hierarchical model.")))
hierarchical_model_scenarios.append(
    pytest.param("CPU", "gradient.allreduce", {}, id="Gradient Allreduce on CPU",
                 marks=pytest.mark.skip(reason="GA may not converge for hierarchical model.")))
hierarchical_model_scenarios.append(
    pytest.param("CPU", "win.put", {'window_prefix': 'CPU'}, id="Window put on CPU",
                 marks=pytest.mark.skip(reason="Win put may not converge for hierarchical model.")))

if TEST_ON_GPU:
    hierarchical_model_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, {"ATC": False},
                     id="AWC Neighbor Allreduce on GPU",
                     marks=pytest.mark.skip(reason="AWC doesn't converge for hierarchical model.")))
    hierarchical_model_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, {"ATC": True},
                     id="ATC Neighbor Allreduce on GPU",
                     marks=pytest.mark.skip(reason="ATC doesn't converge for hierarchical model.")))
    hierarchical_model_scenarios.append(
        pytest.param("GPU", "gradient.allreduce", {}, id="Gradient Allreduce on GPU",
                     marks=pytest.mark.skip(reason="GA may not converge for hierarchical model.")))
    hierarchical_model_scenarios.append(
        pytest.param("GPU", "win.put", {'window_prefix', 'GPU'}, id="Window put on GPU",
                     marks=pytest.mark.skip(
                         reason="Win put may not converge for hierarchical model.")))


@pytest.mark.parametrize("device,communication_type,kwargs", hierarchical_model_scenarios)
def test_optimizer_for_hierarchical_model(device, communication_type, kwargs):
    atc_style = kwargs.get("ATC", False)
    error_threshold = kwargs.get("error_threshold", 1.5)
    window_prefix = kwargs.get("window_prefix", None)

    problem_builder, train_dataloader, test_dataloader, model, optimizer, num_epochs = \
        problem_setup(HierarchicalLinearNet)

    isCUDA = pin_model_to_device(device, model)

    if isinstance(communication_type, bf.CommunicationType):
        base_dist_optimizer = (bf.DistributedAdaptThenCombineOptimizer if atc_style else
                               bf.DistributedAdaptWithCombineOptimizer)
        optimizer = base_dist_optimizer(optimizer, model=model,
                                        communication_type=communication_type)
    elif communication_type == "win.put":
        optimizer = bf.DistributedWinPutOptimizer(optimizer, model=model,
        window_prefix=window_prefix)
    elif communication_type == "gradient.allreduce":
        optimizer = bf.DistributedGradientAllreduceOptimizer(
            optimizer, model=model)
    else:
        raise ValueError("Communication_type under test is not expected.")

    # Train and test
    train_mse = []
    test_mse = []
    for _ in range(num_epochs):
        standard_train(model, optimizer, train_dataloader, isCUDA)
        train_mse.append(evaluation(model, train_dataloader, isCUDA))
        test_mse.append(evaluation(model, test_dataloader, isCUDA))
    train_mse = np.array(train_mse)
    test_mse = np.array(test_mse)

    # Check if the MSEs in the last three epochs are small enough
    assert (
        train_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."
    assert (
        test_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."

    if communication_type == "win.put":
        optimizer.unregister_window()

# Neighbor allreduce dynamic tests
dynamic_neighbor_allreduce_scenarios = []
dynamic_neighbor_allreduce_scenarios.append(
    pytest.param("CPU", False, {}, id="Dynamic AWC Neighbor Allreduce on CPU"))
dynamic_neighbor_allreduce_scenarios.append(
    pytest.param("CPU", True, {}, id="Dynamic ATC Neighbor Allreduce on CPU"))
if TEST_ON_GPU:
    dynamic_neighbor_allreduce_scenarios.append(
        pytest.param("GPU", False, {}, id="Dynamic AWC Neighbor Allreduce on GPU"))
    dynamic_neighbor_allreduce_scenarios.append(
        pytest.param("GPU", True, {}, id="Dynamic ATC Neighbor Allreduce on GPU"))


@pytest.mark.parametrize("device,atc_style,kwargs", dynamic_neighbor_allreduce_scenarios)
def test_dynamic_neighbor_allreduce_optimizer(device, atc_style, kwargs):
    error_threshold = kwargs.get("error_threshold", 1.5)

    problem_builder, train_dataloader, test_dataloader, model, optimizer, num_epochs = \
        problem_setup()

    isCUDA = pin_model_to_device(device, model)

    base_dist_optimizer = (bf.DistributedAdaptThenCombineOptimizer if atc_style else
                           bf.DistributedAdaptWithCombineOptimizer)
    optimizer = base_dist_optimizer(optimizer, model=model,
                                    communication_type=bf.CommunicationType.neighbor_allreduce)

    dynamic_topo_gen = topology_util.GetDynamicOnePeerSendRecvRanks(
        bf.load_topology(), bf.rank())

    # Train and test
    train_mse = []
    test_mse = []
    for _ in range(num_epochs):
        dynamic_neighbor_allreduce_train(model, optimizer, train_dataloader, isCUDA,
                                         dynamic_topo_gen)
        train_mse.append(evaluation(model, train_dataloader, isCUDA))
        test_mse.append(evaluation(model, test_dataloader, isCUDA))
    train_mse = np.array(train_mse)
    test_mse = np.array(test_mse)

    # Check if the MSEs in the last three epochs are small enough
    assert (
        train_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."
    assert (
        test_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."


# Window put dynamic tests
dynamic_win_put_scenarios = []
dynamic_win_put_scenarios.append(
    pytest.param("CPU", {'window_prefix':'CPU'}, id="Dynamic window put on CPU"))
if TEST_ON_GPU:
    dynamic_win_put_scenarios.append(
        pytest.param("GPU", {'window_prefix':'GPU'}, id="Dynamic window put on GPU"))


@pytest.mark.parametrize("device,kwargs", dynamic_win_put_scenarios)
def test_dynamic_win_put_optimizer(device, kwargs):
    error_threshold = kwargs.get("error_threshold", 1.5)
    window_prefix = kwargs.get("window_prefix", None)

    problem_builder, train_dataloader, test_dataloader, model, optimizer, num_epochs = \
        problem_setup()

    isCUDA = pin_model_to_device(device, model)

    optimizer = bf.DistributedWinPutOptimizer(optimizer, model=model, window_prefix=window_prefix)
    
    # Train and test
    train_mse = []
    test_mse = []
    for epoch in range(num_epochs):
        dynamic_win_put_train(
            model, optimizer, train_dataloader, isCUDA, epoch)
        train_mse.append(evaluation(model, train_dataloader, isCUDA))
        test_mse.append(evaluation(model, test_dataloader, isCUDA))
    train_mse = np.array(train_mse)
    test_mse = np.array(test_mse)

    # Check if the MSEs in the last three epochs are small enough
    assert (
        train_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."
    assert (
        test_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."
    optimizer.unregister_window()


local_aggregation_scenarios = []
local_aggregation_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.empty, {"ATC": False, "error_threshold": 2},
                 id="AWC Empty on CPU"))
local_aggregation_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.empty, {"ATC": True, "error_threshold": 2},
                 id="ATC Empty on CPU"))
local_aggregation_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.allreduce, {"ATC": False},
                 id="AWC Allreduce on CPU"))
local_aggregation_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.allreduce, {"ATC": True},
                 id="ATC Allreduce on CPU"))
local_aggregation_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"ATC": False},
                 id="AWC Neighbor Allreduce on CPU"))
local_aggregation_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce,
                 {"ATC": True}, id="ATC Neighbor Allreduce on CPU"))
local_aggregation_scenarios.append(
    pytest.param("CPU", "gradient.allreduce", {}, id="Gradient Allreduce on CPU"))
local_aggregation_scenarios.append(
    pytest.param("CPU", "win.put", {'window_prefix': 'CPU'}, id="Window put on CPU"))
local_aggregation_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"mini_batch_size": 4},
                 id="Neighbor allreduce AWC on CPU with a mini_batch_size of 4"))
local_aggregation_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"mini_batch_size": 8},
                 id="Neighbor allreduce AWC on CPU with a mini_batch_size of 8"))
local_aggregation_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"mini_batch_size": 32},
                 id="Neighbor allreduce AWC on CPU with a mini_batch_size of 32"))
if TEST_ON_GPU:
    local_aggregation_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.empty, {"ATC": False, "error_threshold": 2},
                     id="AWC Empty on GPU"))
    local_aggregation_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.empty, {"ATC": True, "error_threshold": 2},
                     id="ATC Empty on GPU"))
    local_aggregation_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.allreduce, {"ATC": False},
                     id="AWC Allreduce on GPU"))
    local_aggregation_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.allreduce, {"ATC": True},
                     id="ATC Allreduce on GPU"))
    local_aggregation_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, {"ATC": False},
                     id="AWC Neighbor Allreduce on GPU"))
    local_aggregation_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, {"ATC": True},
                     id="ATC Neighbor Allreduce on GPU"))
    local_aggregation_scenarios.append(
        pytest.param("GPU", "gradient.allreduce", {}, id="Gradient Allreduce on GPU"))
    local_aggregation_scenarios.append(
        pytest.param("GPU", "win.put", {'window_prefix': 'GPU'}, id="Window put on GPU"))

@pytest.mark.parametrize("device,communication_type,kwargs", local_aggregation_scenarios)
def test_optimizer_local_aggregation(device, communication_type, kwargs):
    atc_style = kwargs.get("ATC", False)
    error_threshold = kwargs.get("error_threshold", 1.5)
    mini_batch_size = kwargs.get("mini_batch_size", 16)
    window_prefix = kwargs.get("window_prefix", None)

    problem_builder, train_dataloader, test_dataloader, model, optimizer, num_epochs = \
        problem_setup()

    isCUDA = pin_model_to_device(device, model)

    J = train_dataloader.batch_size // mini_batch_size

    if isinstance(communication_type, bf.CommunicationType):
        base_dist_optimizer = (bf.DistributedAdaptThenCombineOptimizer if atc_style else
                               bf.DistributedAdaptWithCombineOptimizer)
        optimizer = base_dist_optimizer(optimizer, model=model,
                                        communication_type=communication_type,
                                        num_steps_per_communication=J)
    elif communication_type == "win.put":
        optimizer = bf.DistributedWinPutOptimizer(optimizer, model=model,
                                                  num_steps_per_communication=J)
    elif communication_type == "gradient.allreduce":
        optimizer = bf.DistributedGradientAllreduceOptimizer(optimizer, model=model,
                                                             num_steps_per_communication=J)
    else:
        raise ValueError("Communication_type under test is not expected.")

    # Train and test
    train_mse = []
    test_mse = []
    for _ in range(num_epochs):
        local_aggregation_train(
            model, optimizer, train_dataloader, isCUDA, mini_batch_size)
        train_mse.append(evaluation(model, train_dataloader, isCUDA))
        test_mse.append(evaluation(model, test_dataloader, isCUDA))
    train_mse = np.array(train_mse)
    test_mse = np.array(test_mse)

    # Check if the MSEs in the last three epochs are small enough
    assert (
        train_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."
    assert (
        test_mse[-3:].max() < error_threshold*problem_builder.noise_level**2
    ), "Train MSE in the last three epochs doesn't coverge."

    if communication_type == "win.put":
        optimizer.unregister_window()

local_aggregation_duplicated_scenarios = []
local_aggregation_duplicated_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"ATC": False},
                 id="AWC Neighbor Allreduce on CPU"))
local_aggregation_duplicated_scenarios.append(
    pytest.param("CPU", bf.CommunicationType.neighbor_allreduce, {"ATC": True},
                 id="ATC Neighbor Allreduce on CPU"))
local_aggregation_duplicated_scenarios.append(
    pytest.param("CPU", "win.put", {'window_prefix': 'CPU'}, id="Win Put on CPU"))
local_aggregation_duplicated_scenarios.append(
    pytest.param("CPU", "gradient.allreduce", {}, id="Gradient Allreduce on CPU"))
if TEST_ON_GPU:
    local_aggregation_duplicated_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, {"ATC": False},
                     id="AWC Neighbor Allreduce on GPU"))
    local_aggregation_duplicated_scenarios.append(
        pytest.param("GPU", bf.CommunicationType.neighbor_allreduce, {"ATC": True},
                     id="ATC Neighbor Allreduce on GPU"))
    local_aggregation_duplicated_scenarios.append(
        pytest.param("GPU", "win.put", {'window_prefix': 'GPU'}, id="Win Put on GPU"))
    local_aggregation_duplicated_scenarios.append(
        pytest.param("GPU", "gradient.allreduce", {}, id="Gradient Allreduce on GPU"))


@pytest.mark.filterwarnings("error:Unexpected behavior")
@pytest.mark.parametrize("device,communication_type,kwargs", local_aggregation_duplicated_scenarios)
def test_optimizer_local_aggregation_duplicated(device, communication_type, kwargs):
    # Accuracy doesn't matter here, mainly to test if there is warning thrown
    # for local aggregation.
    atc_style = kwargs.get("ATC", False)
    mini_batch_size = kwargs.get("mini_batch_size", 16)
    window_prefix = kwargs.get("window_prefix", None)

    _, train_dataloader, test_dataloader, model, optimizer, num_epochs = \
        problem_setup(DuplicatedLinearNet)

    isCUDA = pin_model_to_device(device, model)

    mini_batch_size = train_dataloader.batch_size
    J = train_dataloader.batch_size // mini_batch_size

    if isinstance(communication_type, bf.CommunicationType):
        base_dist_optimizer = (bf.DistributedAdaptThenCombineOptimizer if atc_style else
                               bf.DistributedAdaptWithCombineOptimizer)
        optimizer = base_dist_optimizer(optimizer, model=model,
                                        communication_type=communication_type,
                                        num_steps_per_communication=J)
    elif communication_type == "win.put":
        optimizer = bf.DistributedWinPutOptimizer(optimizer, model=model,
                                                  window_prefix=window_prefix,
                                                  num_steps_per_communication=J)
    elif communication_type == "gradient.allreduce":
        optimizer = bf.DistributedGradientAllreduceOptimizer(optimizer, model=model,
                                                             num_steps_per_communication=J)
    else:
        raise ValueError("Communication_type under test is not expected.")

    # Train and test
    for _ in range(num_epochs):
        local_aggregation_train(
            model, optimizer, train_dataloader, isCUDA, mini_batch_size)
        evaluation(model, train_dataloader, isCUDA)
        evaluation(model, test_dataloader, isCUDA)

    if communication_type == "win.put":
        optimizer.unregister_window()
