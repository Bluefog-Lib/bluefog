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
using ::bluefog::common::Status;

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
  // We need to allocate neighbor_indegree tensor space for it.
  std::vector<std::shared_ptr<TorchTensor>> neighbor_tensors;
  for (int i = 0; i < bluefog_neighbor_size(); i++) {
    auto t = std::make_shared<TorchTensor>(tensor->MakeCopy(device));
    neighbor_tensors.push_back(t);
  }
  tensors_map_[name] = neighbor_tensors;
  return true;
}

bool WinTorchStorageManager::UnregisterWinName(const std::string& name) {
  if (tensors_map_.find(name) == tensors_map_.end()) {
    return false;
  }
  auto it = tensors_map_.find(name);
  tensors_map_.erase(it);
  return true;
}

void WinTorchStorageManager::ClearAll() { tensors_map_.clear(); }

bool WinTorchStorageManager::GetStorageByname(
    const std::string& name, std::vector<std::shared_ptr<common::Tensor>>& tensors) {
  if (tensors_map_.find(name) == tensors_map_.end()) {
    return false;
  }
  auto it = tensors_map_.find(name);
  for (std::shared_ptr<TorchTensor> t : it->second) {
    tensors.emplace_back(t);
  }
  return true;
}

bool WinTorchStorageManager::AvgWithNeighbor(const std::string& name, ::torch::Tensor local_tensor) {
  if (tensors_map_.find(name) == tensors_map_.end()) {
    return false;
  }
  auto it = tensors_map_.find(name);
  for (std::shared_ptr<TorchTensor> t : it->second) {
    local_tensor.add_(t->GetUnderlyingTensor());
  }
  local_tensor.div_(static_cast<int>(it->second.size()) + 1);
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

  if (!win_storage_manager.UnregisterWinName(name)) {
    LOG(ERROR) << "Cannot unregister win " << name;
    return 0;
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

  // TODO(ybc) Add dst_ranks into API
  auto enqueue_result = EnqueuTensorWindowPut(
      bf_tensor, name, device, [handle](const Status& status) {
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

  // TODO(ybc) Add src_ranks into API
  auto enqueue_result = EnqueuTensorWindowGet(
      bf_tensor, name, device, [handle](const Status& status) {
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