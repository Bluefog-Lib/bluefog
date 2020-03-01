#include "mpi_win_ops.h"

#include <torch/extension.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <thread>

#include "../common/logging.h"
#include "../common/operations.h"
#include "adapter.h"
#include "handle_manager.h"

namespace bluefog {
namespace torch {

using ::bluefog::common::bluefog_load_topology;
using ::bluefog::common::bluefog_load_topology_weights;
using ::bluefog::common::bluefog_neighbor_size;
using ::bluefog::common::EnqueuTensorWindowGet;
using ::bluefog::common::Status;
using NeighborTable = std::unordered_map<int, std::shared_ptr<TorchTensor>>;

// static here means Local/private variable.
static HandleManager win_handle_manager;
static WinTorchStorageManager win_storage_manager;

namespace {

int GetDeviceID(const ::torch::Tensor& tensor) {
  if (tensor.device().is_cuda()) {
    return tensor.device().index();
  }
  return CPU_DEVICE_ID;
}

}  // namespace

bool WinTorchStorageManager::RegisterWinName(
    const std::string& name, const int device,
    std::shared_ptr<TorchTensor> tensor) {
  if (tensors_map_.find(name) != tensors_map_.end()) {
    return false;
  }
  int* sources_ptr = nullptr;
  int* destinations_ptr = nullptr;
  bluefog_load_topology(&in_neighbor_degree_, sources_ptr,
                        &out_neighbor_degree_, destinations_ptr);
  // We need to allocate neighbor_indegree tensor space for it.
  NeighborTable neighbor_tensors;
  for (int i = 0; i < in_neighbor_degree_; i++) {
    auto t = std::make_shared<TorchTensor>(tensor->MakeCopy(device));
    int source_rank = *(sources_ptr + i);
    neighbor_tensors[source_rank] = t;
  }
  tensors_map_[name] = neighbor_tensors;
  self_tensor_map_[name] = tensor;
  return true;
}

bool WinTorchStorageManager::UnregisterWinName(const std::string& name) {
  auto it = tensors_map_.find(name);
  if (it == tensors_map_.end()) {
    return false;
  }
  tensors_map_.erase(it);
  self_tensor_map_.erase(self_tensor_map_.find(name));
  return true;
}

void WinTorchStorageManager::ClearAll() {
  tensors_map_.clear();
  self_tensor_map_.clear();
}

bool WinTorchStorageManager::GetStorageByname(
    const std::string& name,
    std::vector<std::shared_ptr<common::Tensor>>& tensors) {
  auto it = tensors_map_.find(name);
  if (it == tensors_map_.end()) {
    return false;
  }
  std::unordered_map<int, std::shared_ptr<TorchTensor>> neighbor_map =
      it->second;
  int* sources_ptr = nullptr;
  int* destinations_ptr = nullptr;
  bluefog_load_topology(&in_neighbor_degree_, sources_ptr,
                        &out_neighbor_degree_, destinations_ptr);

  // We have to insert it in order.
  for (int i = 0; i < in_neighbor_degree_; i++) {
    int source_rank = *(sources_ptr + i);
    tensors.emplace_back(neighbor_map[source_rank]);
  }
  return true;
}

bool WinTorchStorageManager::GetStorageByNameRank(
    const std::string& name, const int rank,
    std::shared_ptr<TorchTensor>& tensor) {
  auto it = tensors_map_.find(name);
  if (it == tensors_map_.end()) {
    BFLOG(ERROR) << "Cannot find " << name << " in neighbor tensor map";
    return false;
  }
  std::unordered_map<int, std::shared_ptr<TorchTensor>> neighbor_map =
      it->second;
  auto it2 = neighbor_map.find(rank);
  if (it2 == neighbor_map.end()) {
    BFLOG(ERROR) << "Cannot find rank " << rank << " in " << name
                 << "neighbor tensor map";
    return false;
  }
  tensor = it2->second;
  return true;
}

bool WinTorchStorageManager::SumWithNeighbor(const std::string& name,
                                             ::torch::Tensor local_tensor) {
  auto it = tensors_map_.find(name);
  if (it == tensors_map_.end()) {
    return false;
  }
  for (auto& kv : it->second) {
    std::shared_ptr<TorchTensor>& t = kv.second;
    local_tensor.add_(t->GetUnderlyingTensor());
  }
  return true;
}

bool WinTorchStorageManager::AvgWithNeighbor(const std::string& name,
                                             ::torch::Tensor local_tensor) {
  const std::unordered_map<int, float>* weights;
  int is_weighted = bluefog_load_topology_weights(weights);
  if (is_weighted == 1) {
    return AvgWithNeighbor(name, local_tensor, *weights);
  } else {
    // By default we use the (uniform) average.
    if (!SumWithNeighbor(name, local_tensor)) {
      return false;
    }
    // +1 here because in neighbor degree doesn't include self rank.
    local_tensor.div_(in_neighbor_degree_ + 1);
    return true;
  }
}

bool WinTorchStorageManager::AvgWithNeighbor(
    const std::string& name, ::torch::Tensor local_tensor,
    const std::unordered_map<int, float>& weights) {
  auto it = tensors_map_.find(name);
  if (it == tensors_map_.end()) {
    return false;
  }

  auto neighbor_map = it->second;
  float self_weight = 0.0;
  if (weights.find(common::bluefog_rank()) != weights.end()) {
    self_weight = static_cast<float>(weights.at(common::bluefog_rank()));
  }
  local_tensor.mul_(self_weight);

  for (auto& kv : weights) {
    int rank = kv.first;
    if (rank == common::bluefog_rank()) continue;
    float weight = kv.second;
    auto neighbor_tesnor = neighbor_map.at(kv.first)->GetUnderlyingTensor();
    local_tensor.add_(neighbor_tesnor.mul_(weight));
  }
  return true;
}

bool WinTorchStorageManager::SumWithNeighbor(
    const std::string& name, ::torch::Tensor local_tensor,
    const std::vector<int>& source_ranks) {
  if (tensors_map_.find(name) == tensors_map_.end()) {
    return false;
  }
  std::unordered_map<int, std::shared_ptr<TorchTensor>> neighbor_map =
      tensors_map_.at(name);
  for (int rank : source_ranks) {
    local_tensor.add_(neighbor_map.at(rank)->GetUnderlyingTensor());
  }
  return true;
}

bool WinTorchStorageManager::AvgWithNeighbor(
    const std::string& name, ::torch::Tensor local_tensor,
    const std::vector<int>& source_ranks) {
  if (!SumWithNeighbor(name, local_tensor, source_ranks)) {
    return false;
  }
  // +1 here because source_ranks doesn't include self rank.
  local_tensor.div_(static_cast<int>(source_ranks.size()) + 1);
  return true;
}

int DoWinCreate(::torch::Tensor tensor, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  std::vector<std::shared_ptr<common::Tensor>> bf_neighbor_tensors;

  if (!win_storage_manager.RegisterWinName(name, device, bf_tensor)) return 0;
  if (!win_storage_manager.GetStorageByname(name, bf_neighbor_tensors))
    return 0;

  Status status = WindowCreate(bf_tensor, bf_neighbor_tensors, name, device);
  return status.ok() ? 1 : 0;
}

namespace {

int InplaceUpdateNeighborTensor(
    const std::string& name,
    const std::unordered_map<int, float>& update_weights) {
  std::shared_ptr<TorchTensor> bf_neighbor_tensor;
  for (auto& kv : update_weights) {
    int rank = kv.first;
    float weight = kv.second;
    if (!win_storage_manager.GetStorageByNameRank(name, rank,
                                                  bf_neighbor_tensor)) {
      BFLOG(FATAL) << "Cannot get neighbor tensor with " << name << " at rank "
                   << rank;
      return 0;
    }
    bf_neighbor_tensor->GetUnderlyingTensor().mul_(weight);
  }
  return 1;
}

}  // namespace

int DoWinSync(::torch::Tensor tensor, const std::string& name,
              const std::unordered_map<int, float>& update_weights) {
  ThrowIfError(common::CheckInitialized());

  Status status = common::WindowSync(name);

  // Averaging with neighbors' tensors happens in-place.
  if (!win_storage_manager.AvgWithNeighbor(name, tensor)) return 0;
  if (!InplaceUpdateNeighborTensor(name, update_weights)) return 0;

  return 1;
}

int DoWinSyncWeighted(::torch::Tensor tensor, const std::string& name,
                      const std::unordered_map<int, float>& weights,
                      const std::unordered_map<int, float>& update_weights) {
  ThrowIfError(common::CheckInitialized());

  Status status = common::WindowSync(name);

  // Weighted averaging with neighbors' tensors happens in-place.
  if (!win_storage_manager.AvgWithNeighbor(name, tensor, weights)) return 0;
  if (!InplaceUpdateNeighborTensor(name, update_weights)) return 0;

  return 1;
}

int DoWinFree(const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  if (name.empty()) {
    win_storage_manager.ClearAll();
  } else {
    auto res = win_storage_manager.UnregisterWinName(name);
    if (!res) {
      BFLOG(ERROR) << "Cannot unregister win " << name;
      return 0;
    }
  }
  Status status = common::WindowFree(name);
  return status.ok() ? 1 : 0;
}

int DoWinPut(::torch::Tensor tensor, const std::string& name,
             const std::unordered_map<int, float>& dst_weights) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  auto handle = win_handle_manager.AllocateHandle();

  auto enqueue_result = EnqueuTensorWindowPut(
      bf_tensor, name, dst_weights, device, [handle](const Status& status) {
        win_handle_manager.MarkDone(handle, status);
      });

  ThrowIfError(enqueue_result);
  return handle;
}

int DoWinAccumulate(::torch::Tensor tensor, const std::string& name,
                    const std::unordered_map<int, float>& dst_weights) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  auto handle = win_handle_manager.AllocateHandle();

  auto enqueue_result = EnqueuTensorWindowAccumulate(
      bf_tensor, name, dst_weights, device, [handle](const Status& status) {
        win_handle_manager.MarkDone(handle, status);
      });

  ThrowIfError(enqueue_result);
  return handle;
}

int DoWinGet(const std::string& name,
             const std::unordered_map<int, float>& src_weights) {
  ThrowIfError(common::CheckInitialized());

  auto handle = win_handle_manager.AllocateHandle();
  auto enqueue_result = EnqueuTensorWindowGet(
      name, src_weights,
      [handle, name, src_weights](const Status& status) mutable {
        std::shared_ptr<TorchTensor> bf_neighbor_tensor;
        for (auto& kv : src_weights) {
          int rank = kv.first;
          float weight = kv.second;
          if (!win_storage_manager.GetStorageByNameRank(name, rank,
                                                        bf_neighbor_tensor)) {
            BFLOG(FATAL) << "Cannot get neighbor tensor with " << name
                         << " at rank " << rank;
          }
          bf_neighbor_tensor->GetUnderlyingTensor().mul_(weight);
        }
        win_handle_manager.MarkDone(handle, status);
      });

  ThrowIfError(enqueue_result);

  return handle;
}

int DoWinPollHandle(int handle) {
  return win_handle_manager.PollHandle(handle) ? 1 : 0;
}

void DoWinWait(int handle) {
  while (!win_handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = win_handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

int DoWinFence(const std::string& name) {
  ThrowIfError(common::CheckInitialized());
  Status status = common::WindowFence(name);
  return status.ok() ? 1 : 0;
}

void AddWinOpsIntoPybind(py::module& m) {
  // one-sided communication
  m.def("bluefog_torch_win_create_torch_IntTensor", &DoWinCreate);
  m.def("bluefog_torch_win_create_torch_LongTensor", &DoWinCreate);
  m.def("bluefog_torch_win_create_torch_FloatTensor", &DoWinCreate);
  m.def("bluefog_torch_win_create_torch_DoubleTensor", &DoWinCreate);
#if HAVE_CUDA
  m.def("bluefog_torch_win_create_torch_cuda_IntTensor", &DoWinCreate);
  m.def("bluefog_torch_win_create_torch_cuda_LongTensor", &DoWinCreate);
  m.def("bluefog_torch_win_create_torch_cuda_FloatTensor", &DoWinCreate);
  m.def("bluefog_torch_win_create_torch_cuda_DoubleTensor", &DoWinCreate);
#endif

  m.def("bluefog_torch_win_sync_torch_IntTensor", &DoWinSync);
  m.def("bluefog_torch_win_sync_torch_LongTensor", &DoWinSync);
  m.def("bluefog_torch_win_sync_torch_FloatTensor", &DoWinSync);
  m.def("bluefog_torch_win_sync_torch_DoubleTensor", &DoWinSync);
#if HAVE_CUDA
  m.def("bluefog_torch_win_sync_torch_cuda_IntTensor", &DoWinSync);
  m.def("bluefog_torch_win_sync_torch_cuda_LongTensor", &DoWinSync);
  m.def("bluefog_torch_win_sync_torch_cuda_FloatTensor", &DoWinSync);
  m.def("bluefog_torch_win_sync_torch_cuda_DoubleTensor", &DoWinSync);
#endif

  m.def("bluefog_torch_win_sync_with_weights_torch_IntTensor",
        &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_LongTensor",
        &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_FloatTensor",
        &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_DoubleTensor",
        &DoWinSyncWeighted);
#if HAVE_CUDA
  m.def("bluefog_torch_win_sync_with_weights_torch_cuda_IntTensor",
        &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_cuda_LongTensor",
        &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_cuda_FloatTensor",
        &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_cuda_DoubleTensor",
        &DoWinSyncWeighted);
#endif

  m.def("bluefog_torch_win_put_torch_IntTensor", &DoWinPut);
  m.def("bluefog_torch_win_put_torch_LongTensor", &DoWinPut);
  m.def("bluefog_torch_win_put_torch_FloatTensor", &DoWinPut);
  m.def("bluefog_torch_win_put_torch_DoubleTensor", &DoWinPut);
#if HAVE_CUDA
  m.def("bluefog_torch_win_put_torch_cuda_IntTensor", &DoWinPut);
  m.def("bluefog_torch_win_put_torch_cuda_LongTensor", &DoWinPut);
  m.def("bluefog_torch_win_put_torch_cuda_FloatTensor", &DoWinPut);
  m.def("bluefog_torch_win_put_torch_cuda_DoubleTensor", &DoWinPut);
#endif

  m.def("bluefog_torch_win_accumulate_torch_IntTensor", &DoWinAccumulate);
  m.def("bluefog_torch_win_accumulate_torch_LongTensor", &DoWinAccumulate);
  m.def("bluefog_torch_win_accumulate_torch_FloatTensor", &DoWinAccumulate);
  m.def("bluefog_torch_win_accumulate_torch_DoubleTensor", &DoWinAccumulate);
#if HAVE_CUDA
  m.def("bluefog_torch_win_accumulate_torch_cuda_IntTensor", &DoWinAccumulate);
  m.def("bluefog_torch_win_accumulate_torch_cuda_LongTensor", &DoWinAccumulate);
  m.def("bluefog_torch_win_accumulate_torch_cuda_FloatTensor",
        &DoWinAccumulate);
  m.def("bluefog_torch_win_accumulate_torch_cuda_DoubleTensor",
        &DoWinAccumulate);
#endif

  m.def("bluefog_torch_win_get", &DoWinGet);

  m.def("bluefog_torch_win_free", &DoWinFree);
  m.def("bluefog_torch_win_fence", &DoWinFence);
  m.def("bluefog_torch_win_poll", &DoWinPollHandle);
  m.def("bluefog_torch_win_wait", &DoWinWait);
}

}  // namespace torch
}  // namespace bluefog