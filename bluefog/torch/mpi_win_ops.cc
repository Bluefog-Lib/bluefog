#include "mpi_win_ops.h"

#include <torch/extension.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

#include "../common/logging.h"
#include "../common/operations.h"
#include "../common/cuda_util.h"
#include "adapter.h"
#include "handle_manager.h"

namespace bluefog {
namespace torch {

using ::bluefog::common::bluefog_load_topology;
using ::bluefog::common::bluefog_load_topology_weights;
using ::bluefog::common::bluefog_neighbor_size;
using ::bluefog::common::with_device;
using ::bluefog::common::Status;
using ::bluefog::common::EnqueueTensorWindowGet;
using NeighborTable = std::unordered_map<int, std::shared_ptr<TorchTensor>>;

// static here means Local/private variable.
static HandleManager win_handle_manager;
static WinTorchStorageManager win_storage_manager;

static const char* BLUEFOG_WIN_ON_CPU = std::getenv("BLUEFOG_WIN_ON_CPU");
static const bool WIN_ON_CPU = (BLUEFOG_WIN_ON_CPU != nullptr) && (*BLUEFOG_WIN_ON_CPU == '1');

// A map store {name -> gpu_tensor}. Used only when BLUEFOG_WIN_ON_CPU is turned on.
static std::unordered_map<std::string, ::torch::Tensor> win_gpu_tensor_map;

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
    std::shared_ptr<TorchTensor> t = kv.second;
    local_tensor.add_(t->GetUnderlyingTensor());
  }
  return true;
}

bool WinTorchStorageManager::AvgWithNeighbor(const std::string& name,
                                             ::torch::Tensor local_tensor) {
  const std::unordered_map<int, float>* weights;
  int is_weighted = bluefog_load_topology_weights(weights);
  if (is_weighted == 1) {
    // TODO(hhb): consider change topology weights definition
    auto it = weights->find(common::bluefog_rank());
    if (it == weights->end()) {
      return false;
    }
    float self_weight = it->second;
    return AvgWithNeighbor(name, local_tensor, self_weight, *weights);
  } else {
    // By default we use the (uniform) average.
    if (!SumWithNeighbor(name, local_tensor)) {
      return false;
    }
    // +1 here because in neighbor degree doesn't include self rank.
    local_tensor.div_(static_cast<float>(in_neighbor_degree_) + 1.0);
    return true;
  }
}

bool WinTorchStorageManager::AvgWithNeighbor(
    const std::string& name, ::torch::Tensor local_tensor,
    float self_weight,
    const std::unordered_map<int, float>& neighbor_weights) {
  auto it = tensors_map_.find(name);
  if (it == tensors_map_.end()) {
    return false;
  }
  
  local_tensor.mul_(self_weight);

  auto neighbor_map = it->second;
  for(auto& kv: weights) {
    int rank = kv.first;
    if(rank == common::bluefog_rank()) continue;
    float weight = kv.second;
    auto neighbor_tesnor = neighbor_map.at(kv.first)->GetUnderlyingTensor();
    local_tensor.add_(neighbor_tesnor.mul(weight));
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
  local_tensor.div_(static_cast<float>(source_ranks.size()) + 1.0);
  return true;
}

int DoWinCreate(::torch::Tensor tensor, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  std::shared_ptr<TorchTensor> bf_tensor;

  if (WIN_ON_CPU && tensor.device().is_cuda()) {
    ::torch::Tensor cpu_buffer =
        tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/true);
    bf_tensor = std::make_shared<TorchTensor>(cpu_buffer);
  } else {
    bf_tensor = std::make_shared<TorchTensor>(tensor);
  }

  std::vector<std::shared_ptr<common::Tensor>> bf_neighbor_tensors;

  if (!win_storage_manager.RegisterWinName(name, device, bf_tensor)) return 0;
  if (!win_storage_manager.GetStorageByname(name, bf_neighbor_tensors))
    return 0;
  if (WIN_ON_CPU && tensor.device().is_cuda()) {
    win_gpu_tensor_map[name] = tensor;
  }

  Status status = WindowCreate(bf_tensor, bf_neighbor_tensors, name, device);
  return status.ok() ? 1 : 0;
}

namespace {

int ResetNeighborTensor(const std::string& name,
        const std::unordered_map<int, float>& neighbor_map) {
  std::shared_ptr<TorchTensor> bf_neighbor_tensor;
  for (auto& kv : neighbor_map) {
    int rank = kv.first;
    if (!win_storage_manager.GetStorageByNameRank(name, rank,
                                                  bf_neighbor_tensor)) {
      BFLOG(FATAL) << "Cannot get neighbor tensor with " << name << " at rank "
                   << rank;
      return 0;
    }
    bf_neighbor_tensor->GetUnderlyingTensor().fill_(0.0);
  }
  return 1;
}

} // namespace

int DoWinSync(::torch::Tensor tensor, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  // We need to lock self avoid updating and win_put/win_accumulate happen at simultaneous time.
  const std::vector<int> self_rank = {common::bluefog_rank()};
  Status status = common::WindowSync(name);

  ::torch::Tensor cpu_buffer = tensor;
  if (WIN_ON_CPU && tensor.device().is_cuda()) {
    cpu_buffer = tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false);
  }

  // Averaging with neighbors' tensors happens in-place.
  if (!win_storage_manager.AvgWithNeighbor(name, cpu_buffer)) return 0;

  if (WIN_ON_CPU && tensor.device().is_cuda()) {
    auto device = GetDeviceID(tensor);
    with_device device_guard(device);
    tensor.copy_(cpu_buffer);
  }

  return 1;
}

int DoWinSyncWeighted(::torch::Tensor tensor, const std::string& name,
                      float self_weight,
                      const std::unordered_map<int, float>& neighbor_weights,
                      bool reset) {
  ThrowIfError(common::CheckInitialized());
  
  // We need to lock self avoid updating and win_put/win_accumulate happen at simultaneous time.
  const std::vector<int> self_rank = {common::bluefog_rank()};
  if (reset && !neighbor_weights.empty()) common::WindowMutexAcquire(self_rank);
  Status status = common::WindowSync(name);

  ::torch::Tensor cpu_buffer = tensor;
  if (WIN_ON_CPU && tensor.device().is_cuda()) {
    cpu_buffer = tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false);
  }

  // Weighted averaging with neighbors' tensors happens in-place.
  auto weights = neighbor_weights;
  weights[common::bluefog_rank()] = self_weight;
  if (!win_storage_manager.AvgWithNeighbor(name, cpu_buffer, weights)) return 0;
  if (reset && !ResetNeighborTensor(name, neighbor_weights)) return 0;

  if (reset && !neighbor_weights.empty()) common::WindowMutexRelease(self_rank);

  if (WIN_ON_CPU && tensor.device().is_cuda()) {
    auto device = GetDeviceID(tensor);
    with_device device_guard(device);
    tensor.copy_(cpu_buffer);
  }

  return 1;
}

int DoWinFree(const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  if (name.empty()) {
    win_storage_manager.ClearAll();
    if (WIN_ON_CPU) win_gpu_tensor_map.clear();
  } else {
    auto res = win_storage_manager.UnregisterWinName(name);
    if (!res) {
      BFLOG(ERROR) << "Cannot unregister win " << name;
      return 0;
    }
    if (WIN_ON_CPU) {
      win_gpu_tensor_map.erase(name);
    }
  }
  Status status = common::WindowFree(name);
  return status.ok() ? 1 : 0;
}

int DoWinPut(::torch::Tensor tensor, const std::string& name,
             const std::unordered_map<int, float>& dst_weights) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto handle = win_handle_manager.AllocateHandle();

  std::shared_ptr<TorchTensor> bf_tensor;

  if (WIN_ON_CPU && tensor.device().is_cuda()) {
    // TODO(ybc) Use non_blocking copy and ready_event to make it faster?
    ::torch::Tensor cpu_buffer =
        tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false);
    bf_tensor = std::make_shared<TorchTensor>(cpu_buffer);
  } else {
    bf_tensor = std::make_shared<TorchTensor>(tensor);
  }

  auto enqueue_result = EnqueueTensorWindowPut(
      bf_tensor, name, dst_weights, device, [handle](const Status& status) {
        win_handle_manager.MarkDone(handle, status);
      });

  ThrowIfError(enqueue_result);
  return handle;
}

int DoWinAccumulate(::torch::Tensor tensor, const std::string& name,
                    const std::unordered_map<int, float>& dst_weights,
                    const bool require_mutex) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto handle = win_handle_manager.AllocateHandle();
  std::shared_ptr<TorchTensor> bf_tensor;

  if (WIN_ON_CPU && tensor.device().is_cuda()) {
    ::torch::Tensor cpu_buffer =
        tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false);
    bf_tensor = std::make_shared<TorchTensor>(cpu_buffer);
  } else {
    bf_tensor = std::make_shared<TorchTensor>(tensor);
  }

  auto enqueue_result = EnqueueTensorWindowAccumulate(
      bf_tensor, name, dst_weights, device, require_mutex,
      [handle](const Status& status) {
        win_handle_manager.MarkDone(handle, status);
      });

  ThrowIfError(enqueue_result);
  return handle;
}

int DoWinGet(const std::string& name,
             const std::unordered_map<int, float>& src_weights) {
  ThrowIfError(common::CheckInitialized());

  auto handle = win_handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorWindowGet(
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
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
  auto status = win_handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

int DoWinFence(const std::string& name) {
  ThrowIfError(common::CheckInitialized());
  Status status = common::WindowFence(name);
  return status.ok() ? 1 : 0;
}

void DoWinLock(const std::string& name) {
  ThrowIfError(common::CheckInitialized());
  Status status = common::WindowLock(name);
  ThrowIfError(status);
}

void DoWinUnlock(const std::string& name) {
  ThrowIfError(common::CheckInitialized());
  Status status = common::WindowUnlock(name);
  ThrowIfError(status);
}

void DoWinMutexAcquire(const std::vector<int>& ranks) {
  ThrowIfError(common::CheckInitialized());
  Status status = common::WindowMutexAcquire(ranks);
  ThrowIfError(status);
}

void DoWinMutexRelease(const std::vector<int>& ranks) {
  ThrowIfError(common::CheckInitialized());
  Status status = common::WindowMutexRelease(ranks);
  ThrowIfError(status);
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

  m.def("bluefog_torch_win_sync_with_weights_torch_IntTensor", &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_LongTensor", &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_FloatTensor", &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_DoubleTensor", &DoWinSyncWeighted);
#if HAVE_CUDA
  m.def("bluefog_torch_win_sync_with_weights_torch_cuda_IntTensor", &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_cuda_LongTensor", &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_cuda_FloatTensor", &DoWinSyncWeighted);
  m.def("bluefog_torch_win_sync_with_weights_torch_cuda_DoubleTensor", &DoWinSyncWeighted);
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
  m.def("bluefog_torch_win_accumulate_torch_cuda_FloatTensor", &DoWinAccumulate);
  m.def("bluefog_torch_win_accumulate_torch_cuda_DoubleTensor", &DoWinAccumulate);
#endif

  m.def("bluefog_torch_win_get", &DoWinGet);

  m.def("bluefog_torch_win_free", &DoWinFree);
  m.def("bluefog_torch_win_fence", &DoWinFence);
  m.def("bluefog_torch_win_poll", &DoWinPollHandle);
  m.def("bluefog_torch_win_wait", &DoWinWait);

  m.def("bluefog_torch_win_lock", &DoWinLock);
  m.def("bluefog_torch_win_unlock", &DoWinUnlock);

  m.def("bluefog_torch_win_mutex_acquire", &DoWinMutexAcquire);
  m.def("bluefog_torch_win_mutex_release", &DoWinMutexRelease);
}

}  // namespace torch
}  // namespace bluefog
