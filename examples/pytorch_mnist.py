# Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import print_function

from bluefog.common import topology_util
import bluefog.torch as bf
import argparse
import os
import sys
import warnings
warnings.simplefilter('ignore')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size", type=int, default=64,
    metavar="N", help="input batch size for training (default: 64)")
parser.add_argument(
    "--test-batch-size", type=int, default=1000,
    metavar="N", help="input batch size for testing (default: 1000)")
parser.add_argument("--epochs", type=int, default=10, metavar="N",
                    help="number of epochs to train (default: 10)")
parser.add_argument(
    "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")
parser.add_argument("--momentum", type=float, default=0.5,
                    metavar="M", help="SGD momentum (default: 0.5)")
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument('--dist-optimizer', type=str, default='neighbor_allreduce',
                    help='The type of distributed optimizer. Supporting options are ' +
                    '[neighbor_allreduce, hierarchical_neighbor_allreduce, allreduce, horovod]')
parser.add_argument('--disable-dynamic-topology', action='store_true',
                    default=False, help=('Disable each iteration to transmit one neighbor ' +
                                         'per iteration dynamically.'))
parser.add_argument('--atc-style', action='store_true', default=False,
                    help='If True, the step of optimizer happened before communication')

parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

if args.dist_optimizer == 'horovod':
    print("importing horovod")
    import horovod.torch as bf

bf.init()

if args.cuda:
    # Bluefog: pin GPU to local rank.
    device_id = bf.local_rank() if bf.nccl_built() else bf.local_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(args.seed)


kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
data_folder_loc = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")
train_dataset = datasets.MNIST(
    os.path.join(data_folder_loc, "data", "data-%d" % bf.rank()),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
# Bluefog: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bf.size(), rank=bf.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
)

test_dataset = datasets.MNIST(
    os.path.join(data_folder_loc, "data", "data-%d" % bf.rank()),
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_sampler = None
# Bluefog: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=bf.size(), rank=bf.rank()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


model = Net()

if args.cuda:
    print("using cuda.")
    # Move model to GPU.
    model.cuda()

# Bluefog: scale learning rate by the number of GPUs.
optimizer = optim.Adam(model.parameters(), lr=args.lr * bf.size())

# Bluefog: broadcast parameters & optimizer state.
bf.broadcast_parameters(model.state_dict(), root_rank=0)
bf.broadcast_optimizer_state(optimizer, root_rank=0)

# Bluefog: wrap optimizer with DistributedOptimizer.
if args.dist_optimizer != 'horovod':
    base_dist_optimizer = (
        bf.DistributedAdaptThenCombineOptimizer if args.atc_style else
        bf.DistributedAdaptWithCombineOptimizer)
if args.dist_optimizer == 'win_put':
    optimizer = bf.DistributedWinPutOptimizer(optimizer, model=model)
elif args.dist_optimizer == 'allreduce':
    optimizer = base_dist_optimizer(
        optimizer, model=model, communication_type=bf.CommunicationType.allreduce)
elif args.dist_optimizer == 'neighbor_allreduce':
    optimizer = base_dist_optimizer(
        optimizer, model=model, communication_type=bf.CommunicationType.neighbor_allreduce)
elif args.dist_optimizer == 'hierarchical_neighbor_allreduce':
    optimizer = base_dist_optimizer(
        optimizer, model=model,
        communication_type=bf.CommunicationType.hierarchical_neighbor_allreduce)
elif args.dist_optimizer == 'gradient_allreduce':
    optimizer = bf.DistributedGradientAllreduceOptimizer(
        optimizer, model=model)
elif args.dist_optimizer == 'horovod':
    optimizer = bf.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters())
else:
    raise ValueError('Unknown args.dist-optimizer type -- ' + args.dist_optimizer + '\n' +
                     'Please set the argument to be one of ' +
                     '[neighbor_allreduce, gradient_allreduce, allreduce, ' +
                     'hierarchical_neighbor_allreduce, win_put, horovod]')

if not args.disable_dynamic_topology and (args.dist_optimizer != 'horovod'):
    if args.dist_optimizer == 'neighbor_allreduce':
        if bf.is_homogeneous() and bf.size() > bf.local_size():
            dynamic_neighbor_allreduce_gen = topology_util.GetInnerOuterExpo2DynamicSendRecvRanks(
                bf.size(),
                local_size=bf.local_size(),
                self_rank=bf.rank())
        else:
            dynamic_neighbor_allreduce_gen = topology_util.GetDynamicOnePeerSendRecvRanks(
                bf.load_topology(), bf.rank())
    elif args.dist_optimizer == 'hierarchical_neighbor_allreduce':
        # This optimizer can use following dynamic topo only so far.
        dynamic_machine_neighbor_allreduce_gen = topology_util.GetExp2DynamicSendRecvMachineRanks(
            world_size=bf.size(),
            local_size=bf.local_size(),
            self_rank=bf.rank(),
            local_rank=bf.local_rank()
        )
    else:
        dynamic_neighbor_allreduce_gen = topology_util.GetDynamicOnePeerSendRecvRanks(
            bf.load_topology(), bf.rank())

def dynamic_topology_update(epoch, batch_idx):
    if args.dist_optimizer == 'win_put':
        if epoch < 3:
            return
        num_out_neighbors = len(bf.out_neighbor_ranks())
        sent_neighbor = bf.out_neighbor_ranks()[batch_idx % num_out_neighbors]
        optimizer.dst_weights = {sent_neighbor: 1.0}
    elif args.dist_optimizer == 'neighbor_allreduce':
        send_neighbors, recv_neighbors = next(dynamic_neighbor_allreduce_gen)
        optimizer.send_neighbors = send_neighbors
        optimizer.neighbor_weights = {r: 1/(len(recv_neighbors) + 1) for r in recv_neighbors}
        optimizer.self_weight = 1 / (len(recv_neighbors) + 1)
        optimizer.enable_topo_check = False
    elif args.dist_optimizer == 'hierarchical_neighbor_allreduce':
        send_machines, recv_machines = next(dynamic_machine_neighbor_allreduce_gen)
        optimizer.send_neighbor_machines = send_machines
        optimizer.neighbor_machine_weights = {r: 1/(len(recv_machines) + 1) for r in recv_machines}
        optimizer.self_weight = 1 / (len(recv_machines) + 1)
        optimizer.enable_topo_check = False
    else:
        pass


def train(epoch):
    model.train()
    # Bluefog: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if not args.disable_dynamic_topology:
            dynamic_topology_update(epoch, batch_idx)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        if args.dist_optimizer == 'neighbor_allreduce' and (batch_idx % 100 == 99):
            optimizer.communication_type = bf.CommunicationType.allreduce

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Bluefog: use train_sampler to determine the number of examples in
            # this worker's partition.
            print(
                "[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    bf.rank(),
                    epoch,
                    batch_idx * len(data),
                    len(train_sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        if args.dist_optimizer == 'neighbor_allreduce' and (batch_idx % 100 == 99):
            optimizer.communication_type = bf.CommunicationType.neighbor_allreduce


def metric_average(val, name):
    tensor = torch.tensor(val)  # pylint: disable=not-callable
    avg_tensor = bf.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(record):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)
                                 ).cpu().float().sum().item()

    # Bluefog: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler) if test_sampler else len(test_dataset)
    test_accuracy /= len(test_sampler) if test_sampler else len(test_dataset)

    # Bluefog: average metric values across workers.
    test_loss = metric_average(test_loss, "avg_loss")
    test_accuracy = metric_average(test_accuracy, "avg_accuracy")

    # Bluefog: print output only on first rank.
    if bf.rank() == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
                test_loss, 100.0 * test_accuracy
            ), flush=True
        )
    record.append((test_loss, 100.0 * test_accuracy))

test_record = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(test_record)
print(f"[{bf.rank()}]: ", test_record)
bf.barrier()
