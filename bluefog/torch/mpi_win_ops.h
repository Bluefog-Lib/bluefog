// Copyright 2020 Bluefog Team. All Rights Reserved.
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

#ifndef BLUEFOG_TORCH_MPI_WIN_OPS_H
#define BLUEFOG_TORCH_MPI_WIN_OPS_H

#include <mutex>
#include <unordered_map>
#include <vector>

#include "../common/common.h"
#include "adapter.h"
#include <TH/TH.h>

#if HAVE_CUDA
#include <THC/THC.h>
#endif

namespace bluefog {
namespace torch {

class WinTorchStorageManager {
 public:
  WinTorchStorageManager() = default;
  WinTorchStorageManager(const WinTorchStorageManager&) = delete;

  // RegisterWinName will do two things.
  // 1. Allocate new tensors space with the number of in-neighbor copies.
  // 2. Those new tensors will be managed by shared_ptr and pushed into
  // tensors_map_, which use name as the key.
  bool RegisterWinName(const std::string& name, int device,
                       std::shared_ptr<TorchTensor> tensor,
                       const bool zero_init);
  
  // Pop the coresponding tnesors out of tensors_map_ and allocated memory
  // of torch tensor should be destroyed here.
  bool UnregisterWinName(const std::string& name);
  
  // Make the reference tensors point to the neighbor tensor location.
  // The vector contains the neighbor tensors only with same order as
  // bluefog_load_topology return the in_neighbor order.
  bool GetStorageByname(const std::string& name,
                        std::vector<std::shared_ptr<common::Tensor>>& tensors);

  // Make the reference tensor point to the neighbor tensor.
  bool GetStorageByNameRank(const std::string& name, const int rank,
                            std::shared_ptr<TorchTensor>& tensor);

  // Get the device associated with registered name.
  bool GetDeviceByName(const std::string& name, int* device);

  // Sum the local tensor with all neighbor tensors.
  bool SumWithNeighbor(const std::string& name, ::torch::Tensor local_tensor);
  
  // Average the local tensor with neighbor tensors
  // If the weights is set in the mpi_context class, then weighted average will
  // be executed.
  bool AvgWithNeighbor(const std::string& name, ::torch::Tensor local_tensor);

  // Weighted Average the local tensor with neighbor tensors according to weights map.
  // Weights map { rank: weights }. Rank has to be (in-)neighbor ranks. self_weight
  // specifies the weight for self rank.
  // The sum weights are not necessary to be 1.
  // No matter the weights in the mpi_context class is set or not, weights provided in
  // the argument will override it.
  bool AvgWithNeighbor(
      const std::string& name, ::torch::Tensor local_tensor,
      float self_weight,
      const std::unordered_map<int, float>& neighbor_weights);

  
  // This is just utility functions and never used the weights defined in the
  // the mpi_context.
  bool SumWithNeighbor(
      const std::string& name, ::torch::Tensor local_tensor,
      const std::vector<int>& source_ranks);
  bool AvgWithNeighbor(
      const std::string& name, ::torch::Tensor local_tensor,
      const std::vector<int>& source_ranks);
  
  // Clear all storage/reference to neighbor TorchTensor.
  void ClearAll();

 private:
  // A local storage for neighbor's remote memory access.
  // All tensors are associated with an unique name and duplicated for the
  // number of in-degree neighbors.
  // { Tensor Name ->  {rank : tensor } }
  std::unordered_map<std::string,
                     std::unordered_map<int, std::shared_ptr<TorchTensor>>>
      tensors_map_;
  
  std::unordered_map<std::string, std::shared_ptr<TorchTensor>> self_tensor_map_;

  std::unordered_map<std::string, int> device_map_;

  mutable std::mutex mutex_;
  int in_neighbor_degree_;
  int out_neighbor_degree_;
};

#define WIN_CREATE_H(torch_Tensor, THTensor)                     \
  extern "C" int bluefog_torch_win_create_##torch_Tensor(        \
      THTensor* tensor, char* name, bool zero_init);

WIN_CREATE_H(torch_IntTensor, THIntTensor)
WIN_CREATE_H(torch_LongTensor, THLongTensor)
WIN_CREATE_H(torch_FloatTensor, THFloatTensor)
WIN_CREATE_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
WIN_CREATE_H(torch_cuda_IntTensor, THCudaIntTensor)
WIN_CREATE_H(torch_cuda_LongTensor, THCudaLongTensor)
WIN_CREATE_H(torch_cuda_FloatTensor, THCudaTensor)
WIN_CREATE_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define WIN_SYNC_H(torch_Tensor, THTensor)                     \
  extern "C" int bluefog_torch_win_sync_##torch_Tensor(        \
      THTensor* tensor, char* name,                            \
      float self_weight,                                       \
      const std::unordered_map<int, float>& neighbor_weights,  \
      bool reset, bool internal_avg);

WIN_SYNC_H(torch_IntTensor, THIntTensor)
WIN_SYNC_H(torch_LongTensor, THLongTensor)
WIN_SYNC_H(torch_FloatTensor, THFloatTensor)
WIN_SYNC_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
WIN_SYNC_H(torch_cuda_IntTensor, THCudaIntTensor)
WIN_SYNC_H(torch_cuda_LongTensor, THCudaLongTensor)
WIN_SYNC_H(torch_cuda_FloatTensor, THCudaTensor)
WIN_SYNC_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define WIN_PUT_H(torch_Tensor, THTensor)                             \
  extern "C" int bluefog_torch_win_put_##torch_Tensor(                \
      THTensor* tensor, char* name,                                   \
      const std::unordered_map<int, float>& dst_weights);

WIN_PUT_H(torch_IntTensor, THIntTensor)
WIN_PUT_H(torch_LongTensor, THLongTensor)
WIN_PUT_H(torch_FloatTensor, THFloatTensor)
WIN_PUT_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
WIN_PUT_H(torch_cuda_IntTensor, THCudaIntTensor)
WIN_PUT_H(torch_cuda_LongTensor, THCudaLongTensor)
WIN_PUT_H(torch_cuda_FloatTensor, THCudaTensor)
WIN_PUT_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define WIN_ACCUMULATE_H(torch_Tensor, THTensor)                         \
  extern "C" int bluefog_torch_win_accumulate_##torch_Tensor(            \
      THTensor* tensor, char* name,                                      \
      const std::unordered_map<int, float>& dst_weights,                 \
      const bool require_mutex);

WIN_ACCUMULATE_H(torch_IntTensor, THIntTensor)
WIN_ACCUMULATE_H(torch_LongTensor, THLongTensor)
WIN_ACCUMULATE_H(torch_FloatTensor, THFloatTensor)
WIN_ACCUMULATE_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
WIN_ACCUMULATE_H(torch_cuda_IntTensor, THCudaIntTensor)
WIN_ACCUMULATE_H(torch_cuda_LongTensor, THCudaLongTensor)
WIN_ACCUMULATE_H(torch_cuda_FloatTensor, THCudaTensor)
WIN_ACCUMULATE_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

extern "C" int bluefog_torch_win_GET(
    char* name, const std::unordered_map<int, float>& src_weights);

extern "C" int bluefog_torch_win_free(char* name);
extern "C" int bluefog_torch_win_fence(char* name);
extern "C" int bluefog_torch_win_poll(int handle);
extern "C" void bluefog_torch_win_wait(int handle);

extern "C" void bluefog_torch_win_lock(char* name);
extern "C" void bluefog_torch_win_unlock(char* name);

extern "C" void bluefog_torch_win_mutex_acquire(const std::vector<int>& ranks);
extern "C" void bluefog_torch_win_mutex_release(const std::vector<int>& ranks);

}  // namespace torch
}  // namespace bluefog

#endif  // BLUEFOG_TORCH_MPI_WIN_OPS_H
