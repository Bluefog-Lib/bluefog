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

  bool RegisterWinName(const std::string& name, int device,
                       std::shared_ptr<TorchTensor> tensor);
  bool UnregisterWinName(const std::string& name);
  bool GetStorageByname(const std::string& name,
                        std::vector<std::shared_ptr<common::Tensor>>& tensors);
  bool AvgWithNeighbor(const std::string& name, ::torch::Tensor local_tensor);
  bool SumWithNeighbor(const std::string& name, ::torch::Tensor local_tensor);
  void ClearAll();

 private:
  // A local storage for neighbor's remote memory access.
  // All tensors are associated with an unique name and duplicated for the
  // number of in-degree neighbors.
  std::unordered_map<std::string, std::vector<std::shared_ptr<TorchTensor>>>
      tensors_map_;

  mutable std::mutex mutex_;
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

}  // namespace torch
}  // namespace bluefog

#endif  // BLUEFOG_TORCH_MPI_WIN_OPS_H