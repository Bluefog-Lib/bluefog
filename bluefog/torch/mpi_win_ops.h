#ifndef BLUEFOG_TORCH_MPI_WIN_OPS_H
#define BLUEFOG_TORCH_MPI_WIN_OPS_H

#include <mutex>
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
                       std::shared_ptr<TorchTensor> tensor);
  
  // Pop the coresponding tnesors out of tensors_map_ and allocated memory
  // of torch tensor should be destroyed here.
  bool UnregisterWinName(const std::string& name);
  
  // Make the reference tensors point to the neighbor tensor location.
  bool GetStorageByname(const std::string& name,
                        std::vector<std::shared_ptr<common::Tensor>>& tensors);
  
  // Sum the local tensor with the neighbor tensors. If source ranks are not
  // provided it will sum up all the neighbor tensors.
  bool SumWithNeighbor(const std::string& name, ::torch::Tensor local_tensor);
  bool SumWithNeighbor(
      const std::string& name, ::torch::Tensor local_tensor,
      const std::vector<int>& source_ranks);

  // Same as SumWithNeighbor except we will divided by the number of in_neighbor
  // or the size of provided source_ranks.
  bool AvgWithNeighbor(const std::string& name, ::torch::Tensor local_tensor);
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

  mutable std::mutex mutex_;
  int in_neighbor_degree_;
  int out_neighbor_degree_;
};

#define WIN_CREATE_H(torch_Tensor, THTensor)                     \
  extern "C" int bluefog_torch_win_create_##torch_Tensor(        \
      THTensor* tensor, char* name);

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
      THTensor* tensor, char* name);

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

#define WIN_PUT_H(torch_Tensor, THTensor)              \
  extern "C" int bluefog_torch_win_put_##torch_Tensor( \
      THTensor* tensor, char* name, const std::vector<int>& dst_ranks);

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

#define WIN_GET_H(torch_Tensor, THTensor)                              \
  extern "C" int bluefog_torch_win_GET_##torch_Tensor(                 \
      THTensor* tensor, char* name, const std::vector<int>& src_ranks, \
      bool average);

WIN_GET_H(torch_IntTensor, THIntTensor)
WIN_GET_H(torch_LongTensor, THLongTensor)
WIN_GET_H(torch_FloatTensor, THFloatTensor)
WIN_GET_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
WIN_GET_H(torch_cuda_IntTensor, THCudaIntTensor)
WIN_GET_H(torch_cuda_LongTensor, THCudaLongTensor)
WIN_GET_H(torch_cuda_FloatTensor, THCudaTensor)
WIN_GET_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

extern "C" int bluefog_torch_win_free(char* name);
extern "C" int bluefog_torch_win_poll(int handle);
extern "C" void bluefog_torch_win_wait(int handle);
extern "C" int bluefog_torch_win_fence(char* name);

}  // namespace torch
}  // namespace bluefog

#endif  // BLUEFOG_TORCH_MPI_WIN_OPS_H