// Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef BLUEFOG_TORCH_MPI_OPS_H
#define BLUEFOG_TORCH_MPI_OPS_H

#include <unordered_map>

#include "../common/common.h"
#include <TH/TH.h>

#if HAVE_CUDA
#include <THC/THC.h>
#endif

namespace bluefog {
namespace torch {

#define ALLREDUCE_H(torch_Tensor, THTensor)                                    \
  extern "C" int bluefog_torch_allreduce_nonblocking_##torch_Tensor(           \
      THTensor* tensor, THTensor* output, int average,                         \
      bool is_hierarchical_local, char* name);

ALLREDUCE_H(torch_IntTensor, THIntTensor)
ALLREDUCE_H(torch_LongTensor, THLongTensor)
ALLREDUCE_H(torch_HalfTensor, THHalfTensor)
ALLREDUCE_H(torch_FloatTensor, THFloatTensor)
ALLREDUCE_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
ALLREDUCE_H(torch_cuda_IntTensor, THCudaIntTensor)
ALLREDUCE_H(torch_cuda_LongTensor, THCudaLongTensor)
ALLREDUCE_H(torch_cuda_HalfTensor, THCudaHalfTensor)
ALLREDUCE_H(torch_cuda_FloatTensor, THCudaTensor)
ALLREDUCE_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define BROADCAST_H(torch_Tensor, THTensor)                                     \
  extern "C" int bluefog_torch_broadcast_nonblocking_##torch_Tensor(            \
      THTensor* tensor, THTensor* output, int root_rank, char* name);

BROADCAST_H(torch_ByteTensor, THByteTensor)
BROADCAST_H(torch_CharTensor, THCharTensor)
BROADCAST_H(torch_ShortTensor, THShortTensor)
BROADCAST_H(torch_IntTensor, THIntTensor)
BROADCAST_H(torch_LongTensor, THLongTensor)
BROADCAST_H(torch_HalfTensor, THHalfTensor)
BROADCAST_H(torch_FloatTensor, THFloatTensor)
BROADCAST_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
BROADCAST_H(torch_cuda_IntTensor, THCudaIntTensor)
BROADCAST_H(torch_cuda_LongTensor, THCudaLongTensor)
BROADCAST_H(torch_cuda_HalfTensor, THCudaHalfTensor)
BROADCAST_H(torch_cuda_FloatTensor, THCudaTensor)
BROADCAST_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define ALLGATHER_H(torch_Tensor, THTensor)                                    \
  extern "C" int bluefog_torch_allgather_nonblocking_##torch_Tensor(           \
      THTensor* tensor, THTensor* output, char* name);

ALLGATHER_H(torch_ByteTensor, THByteTensor)
ALLGATHER_H(torch_CharTensor, THCharTensor)
ALLGATHER_H(torch_ShortTensor, THShortTensor)
ALLGATHER_H(torch_IntTensor, THIntTensor)
ALLGATHER_H(torch_LongTensor, THLongTensor)
ALLGATHER_H(torch_HalfTensor, THHalfTensor)
ALLGATHER_H(torch_FloatTensor, THFloatTensor)
ALLGATHER_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
ALLGATHER_H(torch_cuda_IntTensor, THCudaIntTensor)
ALLGATHER_H(torch_cuda_LongTensor, THCudaLongTensor)
ALLGATHER_H(torch_cuda_HalfTensor, THCudaHalfTensor)
ALLGATHER_H(torch_cuda_FloatTensor, THCudaTensor)
ALLGATHER_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define NEIGHBOR_ALLGATHER_H(torch_Tensor, THTensor)                              \
  extern "C" int bluefog_torch_neighbor_allgather_nonblocking_##torch_Tensor(     \
      THTensor* tensor, THTensor* output, char* name);

NEIGHBOR_ALLGATHER_H(torch_ByteTensor, THByteTensor)
NEIGHBOR_ALLGATHER_H(torch_CharTensor, THCharTensor)
NEIGHBOR_ALLGATHER_H(torch_ShortTensor, THShortTensor)
NEIGHBOR_ALLGATHER_H(torch_IntTensor, THIntTensor)
NEIGHBOR_ALLGATHER_H(torch_LongTensor, THLongTensor)
NEIGHBOR_ALLGATHER_H(torch_HalfTensor, THHalfTensor)
NEIGHBOR_ALLGATHER_H(torch_FloatTensor, THFloatTensor)
NEIGHBOR_ALLGATHER_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
NEIGHBOR_ALLGATHER_H(torch_cuda_IntTensor, THCudaIntTensor)
NEIGHBOR_ALLGATHER_H(torch_cuda_LongTensor, THCudaLongTensor)
NEIGHBOR_ALLGATHER_H(torch_cuda_HalfTensor, THCudaHalfTensor)
NEIGHBOR_ALLGATHER_H(torch_cuda_FloatTensor, THCudaTensor)
NEIGHBOR_ALLGATHER_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define NEIGHBOR_ALLREDUCE_H(torch_Tensor, THTensor)                             \
  extern "C" int bluefog_torch_neighbor_allreduce_nonblocking_##torch_Tensor(    \
      THTensor* tensor, THTensor* output, double self_weight,                    \
      const std::unordered_map<int, double>& neighbor_weights,                   \
      const std::vector<int>& send_neighbors, bool dynamic_neighbors_enabled,    \
      bool enable_topo_check, bool avg_computation, bool is_hierarchical,        \
      char* name);

NEIGHBOR_ALLREDUCE_H(torch_HalfTensor, THHalfTensor)
NEIGHBOR_ALLREDUCE_H(torch_FloatTensor, THFloatTensor)
NEIGHBOR_ALLREDUCE_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
NEIGHBOR_ALLREDUCE_H(torch_cuda_HalfTensor, THCudaHalfTensor)
NEIGHBOR_ALLREDUCE_H(torch_cuda_FloatTensor, THCudaTensor)
NEIGHBOR_ALLREDUCE_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define PAIR_GOSSIP_H(torch_Tensor, THTensor)                             \
  extern "C" int bluefog_torch_pair_gossip_nonblocking_##torch_Tensor(    \
      THTensor* tensor, THTensor* output, int target_rank,                \
      double self_weight,  double pair_weight,  bool avg_computation,     \
      char* name);

PAIR_GOSSIP_H(torch_HalfTensor, THHalfTensor)
PAIR_GOSSIP_H(torch_FloatTensor, THFloatTensor)
PAIR_GOSSIP_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
PAIR_GOSSIP_H(torch_cuda_HalfTensor, THCudaHalfTensor)
PAIR_GOSSIP_H(torch_cuda_FloatTensor, THCudaTensor)
PAIR_GOSSIP_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

extern "C" int bluefog_torch_poll(int handle);
extern "C" void bluefog_torch_wait_and_clear(int handle);
extern "C" void bluefog_torch_barrier();

}  // namespace torch
}  // namespace bluefog

#endif  // BLUEFOG_TORCH_MPI_OPS_H
