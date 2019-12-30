#include "mpi_win_ops.h"

#include <torch/extension.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <thread>

#include "../common/operations.h"
#include "adapter.h"
#include "handle_manager.h"

namespace bluefog {
namespace torch {

using ::bluefog::common::bluefog_neighbor_size;
using ::bluefog::common::bluefog_load_topology;
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

bool WinTorchStorageManager::RegisterWinName(const std::string& name,
                                        const int device,
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
  return true;
}

bool WinTorchStorageManager::UnregisterWinName(const std::string& name) {
  auto it = tensors_map_.find(name);
  if (it == tensors_map_.end()) {
    return false;
  }
  tensors_map_.erase(it);
  return true;
}

void WinTorchStorageManager::ClearAll() { tensors_map_.clear(); }

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

bool WinTorchStorageManager::AvgWithNeighbor(const std::string& name,
                                             ::torch::Tensor local_tensor) {
  if (!SumWithNeighbor(name, local_tensor)) {
    return false;
  }
  // +1 here because in neighbor degree doesn't include self rank.
  local_tensor.div_(in_neighbor_degree_ + 1);
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

int DoWinSync(::torch::Tensor tensor, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  Status status = common::WindowSync(name);

  // Average with neighbors happens in-place.
  if (!win_storage_manager.AvgWithNeighbor(name, tensor)) return 0;

  return 1;
}

int DoWinFree(const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  if (name.empty()) {
    win_storage_manager.ClearAll();
  } else {
    auto res = win_storage_manager.UnregisterWinName(name);
    if (!res) {
      LOG(ERROR) << "Cannot unregister win " << name;
      return 0;
    }
  }
  Status status = common::WindowFree(name);
  return status.ok() ? 1 : 0;
}

int DoWinPut(::torch::Tensor tensor, const std::string& name,
             const std::vector<int>& dst_ranks) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  auto handle = win_handle_manager.AllocateHandle();

  auto enqueue_result = EnqueuTensorWindowPut(
      bf_tensor, name, dst_ranks, device, [handle](const Status& status) {
        win_handle_manager.MarkDone(handle, status);
      });

  ThrowIfError(enqueue_result);
  return handle;
}

int DoWinGet(::torch::Tensor tensor, const std::string& name,
             const std::vector<int>& src_ranks, bool average) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  auto handle = win_handle_manager.AllocateHandle();

  auto enqueue_result = EnqueuTensorWindowGet(
      bf_tensor, name, src_ranks, device,
      [tensor, name, src_ranks, handle, average](const Status& status) mutable {
        win_storage_manager.SumWithNeighbor(name, tensor, src_ranks);
        if (average) {
          tensor.div_(static_cast<int>(src_ranks.size() + 1));
        }
        win_handle_manager.MarkDone(handle, status);
      });

  ThrowIfError(enqueue_result);

  return handle;
}

int WinPollHandle(int handle) {
  return win_handle_manager.PollHandle(handle) ? 1 : 0;
}

void WinWait(int handle) {
  while (!win_handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = win_handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
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

  m.def("bluefog_torch_win_get_torch_IntTensor", &DoWinGet);
  m.def("bluefog_torch_win_get_torch_LongTensor", &DoWinGet);
  m.def("bluefog_torch_win_get_torch_FloatTensor", &DoWinGet);
  m.def("bluefog_torch_win_get_torch_DoubleTensor", &DoWinGet);
#if HAVE_CUDA
  m.def("bluefog_torch_win_get_torch_cuda_IntTensor", &DoWinGet);
  m.def("bluefog_torch_win_get_torch_cuda_LongTensor", &DoWinGet);
  m.def("bluefog_torch_win_get_torch_cuda_FloatTensor", &DoWinGet);
  m.def("bluefog_torch_win_get_torch_cuda_DoubleTensor", &DoWinGet);
#endif

  m.def("bluefog_torch_win_free", &DoWinFree);
  m.def("bluefog_torch_win_poll", &WinPollHandle);
  m.def("bluefog_torch_win_wait", &WinWait);
}

}  // namespace torch
}  // namespace bluefog