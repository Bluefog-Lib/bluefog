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

import argparse
import os
import sys
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import tensorboardX
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

cwd_folder_loc = os.path.dirname(os.path.abspath(__file__))
# Training settings
parser = argparse.ArgumentParser(
    description="PyTorch ImageNet Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--log-dir",
    default=os.path.join(cwd_folder_loc, "logs"),
    help="tensorboard log directory",
)
parser.add_argument(
    "--checkpoint-format",
    default=os.path.join(
        cwd_folder_loc, "checkpoint", "cifar10-checkpoint-{epoch}.pth.tar"
    ),
    help="checkpoint file format",
)
parser.add_argument(
    "--batches-per-allreduce",
    type=int,
    default=1,
    help="number of batches processed locally before "
    "executing allreduce across workers; it multiplies "
    "total batch size.",
)

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument(
    "--batch-size", type=int, default=32, help="input batch size for training"
)
parser.add_argument(
    "--val-batch-size", type=int, default=32, help="input batch size for validation"
)
parser.add_argument("--epochs", type=int, default=50,
                    help="number of epochs to train")
parser.add_argument(
    "--base-lr", type=float, default=0.0125, help="learning rate for a single GPU"
)
parser.add_argument(
    "--warmup-epochs", type=float, default=5, help="number of warmup epochs"
)
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--wd", type=float, default=0.00005, help="weight decay")

parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--no-bluefog", action="store_true",
                    default=False, help="disables bluefog library")
parser.add_argument("--no-rma", action="store_true",
                    default=False, help="Do no use remote memory access(no window ops).")

args = parser.parse_args()
args.cuda = (not args.no_cuda) and (torch.cuda.is_available())
args.bluefog = not args.no_bluefog

allreduce_batch_size = args.batch_size * args.batches_per_allreduce

# Bluefog: initialize library.
if args.bluefog:
    print("importing bluefog")
    import bluefog.torch as bf
    from bluefog.common import topology_util
else:
    print("importing horovod")
    import horovod.torch as bf

bf.init()

if args.bluefog:
    bf.set_topology(topology=topology_util.PowerTwoRingGraph(bf.size()))

torch.manual_seed(args.seed)

if args.cuda:
    print("using cuda.")
    # Bluefog: pin GPU to local rank.
    torch.cuda.set_device(bf.local_rank() % torch.cuda.device_count())
    torch.cuda.manual_seed(args.seed)
else:
    print("using cpu")

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
# for try_epoch in range(args.epochs, 0, -1):
#     if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
#         resume_from_epoch = try_epoch
#         break

# Bluefog: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = bf.broadcast(
    torch.tensor(resume_from_epoch),  # pylint: disable=not-callable
    root_rank=0,
    name="resume_from_epoch",
).item()

# Bluefog: print logs on the first worker.
verbose = 1 if bf.rank() == 0 else 0

# Bluefog: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(
    args.log_dir) if bf.rank() == 0 else None


kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
train_dataset = datasets.CIFAR10(
    os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    ),
)

# Bluefog: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=bf.size()` and `rank=bf.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bf.size(), rank=bf.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size, sampler=train_sampler, **kwargs
)

val_dataset = datasets.CIFAR10(
    os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    ),
)
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=bf.size(), rank=bf.rank()
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.val_batch_size, sampler=val_sampler, **kwargs
)

# Set up standard ResNet-18 model.
model = models.resnet18()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Bluefog: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(
    model.parameters(),
    lr=(args.base_lr * args.batches_per_allreduce * bf.size()),
    momentum=args.momentum,
    weight_decay=args.wd,
)

# Bluefog: wrap optimizer with DistributedOptimizer.
if args.bluefog:
    if args.no_rma:
        print("Use neighbor collective")
        # This distributed optimizer uses neighbor communication.
        optimizer = bf.DistributedNeighborAllreduceOptimizer(
            optimizer, named_parameters=model.named_parameters()
        )
        if os.environ.get("BLUEFOG_TIMELINE"):
            print("Timeline for optimizer is enabled")
            optimizer.turn_on_timeline(model)
    else:
        # This distributed optimizer uses one-sided communication
        print("Use win_put ops.")
        optimizer = bf.DistributedBluefogOptimizer(
            optimizer, model=model
        )
        if os.environ.get("BLUEFOG_TIMELINE"):
            print("Timeline for optimizer is enabled")
            optimizer.turn_on_timeline(model)
else:
    optimizer = bf.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

print("resume_from_epoch: ", resume_from_epoch)
# Restore from a previous checkpoint, if initial_epoch is specified.
# Bluefog: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and bf.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# Bluefog: broadcast parameters & optimizer state.
bf.broadcast_parameters(model.state_dict(), root_rank=0)
bf.broadcast_optimizer_state(optimizer, root_rank=0)


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric("train_loss")
    train_accuracy = Metric("train_accuracy")

    with tqdm(total=len(train_loader), desc="Train Epoch     #{}".format(epoch + 1),
              disable=not verbose,) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i: i + args.batch_size]
                target_batch = target[i: i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix(
                {
                    "loss": train_loss.avg.item(),
                    "accuracy": 100.0 * train_accuracy.avg.item(),
                }
            )
            t.update(1)

    if log_writer:
        log_writer.add_scalar("train/loss", train_loss.avg, epoch)
        log_writer.add_scalar("train/accuracy", train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric("val_loss")
    val_accuracy = Metric("val_accuracy")

    with tqdm(total=len(val_loader), desc="Validate Epoch  #{}".format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "accuracy": 100.0 * val_accuracy.avg.item(),
                    }
                )
                t.update(1)

    if log_writer:
        log_writer.add_scalar("val/loss", val_loss.avg, epoch)
        log_writer.add_scalar("val/accuracy", val_accuracy.avg, epoch)


# Bluefog: using `lr = base_lr * bf.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * bf.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1.0 / bf.size() * (epoch * (bf.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.0
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group["lr"] = (
            args.base_lr * bf.size() * args.batches_per_allreduce * lr_adj
        )


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if bf.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        state = {"model": model.state_dict(
        ), "optimizer": optimizer.state_dict()}
        torch.save(state, filepath)


# Bluefog: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.0)  # pylint: disable=not-callable
        self.n = torch.tensor(0.0)  # pylint: disable=not-callable

    def update(self, val):
        self.sum += bf.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    validate(epoch)
    save_checkpoint(epoch)
