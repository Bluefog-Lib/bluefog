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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import torch
from bluefog.common.util import check_extension
from bluefog.torch.optimizers import (
    DistributedAllreduceOptimizer,
    DistributedNeighborAllreduceOptimizer,
    DistributedPushSumOptimizer,
    DistributedPullGetOptimizer,
    DistributedBluefogOptimizer)

check_extension('bluefog.torch', __file__, 'mpi_lib')

from bluefog.torch.mpi_ops import init, shutdown
from bluefog.torch.mpi_ops import size, local_size, rank, local_rank
from bluefog.torch.mpi_ops import load_topology, set_topology
from bluefog.torch.mpi_ops import in_neighbor_ranks, out_neighbor_ranks
from bluefog.torch.mpi_ops import mpi_threads_supported
from bluefog.torch.mpi_ops import unified_mpi_window_model_supported
from bluefog.torch.mpi_ops import nccl_built

from bluefog.torch.mpi_ops import allreduce, allreduce_nonblocking
from bluefog.torch.mpi_ops import allgather, allgather_nonblocking
from bluefog.torch.mpi_ops import broadcast, broadcast_nonblocking
from bluefog.torch.mpi_ops import broadcast_, broadcast_nonblocking_
from bluefog.torch.mpi_ops import neighbor_allgather, neighbor_allgather_nonblocking
from bluefog.torch.mpi_ops import neighbor_allreduce, neighbor_allreduce_nonblocking
from bluefog.torch.mpi_ops import poll, synchronize, barrier

from bluefog.torch.mpi_ops import win_create, win_free
from bluefog.torch.mpi_ops import win_update, win_update_then_collect
from bluefog.torch.mpi_ops import win_put_nonblocking, win_put
from bluefog.torch.mpi_ops import win_get_nonblocking, win_get
from bluefog.torch.mpi_ops import win_accumulate_nonblocking, win_accumulate
from bluefog.torch.mpi_ops import win_wait, win_poll
from bluefog.torch.mpi_ops import win_lock, win_mutex

from bluefog.torch.mpi_ops import timeline_start_activity, timeline_end_activity
from bluefog.torch.mpi_ops import timeline_context
from bluefog.torch.utility import broadcast_optimizer_state, broadcast_parameters