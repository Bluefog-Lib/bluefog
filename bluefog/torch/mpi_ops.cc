#include <chrono>
#include <memory>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>

#include "adapter.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "../common/operations.h"

namespace bluefog {
namespace torch {

using ::bluefog::common::bluefog_neighbor_size;
using ::bluefog::common::bluefog_rank;
using ::bluefog::common::bluefog_size;
using ::bluefog::common::Status;

// static here means Local/private variable.
static HandleManager handle_manager;

namespace {

std::string GetOpName(const std::string& prefix, const std::string& name,
                      int handle) {
  if (!name.empty()) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

int GetDeviceID(const ::torch::Tensor& tensor) {
  if (tensor.device().is_cuda()) {
    return tensor.device().index();
  }
  return CPU_DEVICE_ID;
}

}  // namespace

int DoAllreduce(::torch::Tensor tensor, ::torch::Tensor output, int average,
                const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  auto bf_output = std::make_shared<TorchTensor>(output);

  auto enqueue_result = EnqueueTensorAllreduce(
      bf_tensor, bf_output, GetOpName("allreduce", name, handle), device,
      [handle, average, output](const Status& status) mutable {
        // Will execute in the `device` context.
        if (average) {
          output.div_(bluefog_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoBroadcast(::torch::Tensor tensor, ::torch::Tensor output, int root_rank,
                const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  std::shared_ptr<common::Tensor> bf_output = nullptr;
  if (bluefog_rank() == root_rank) {
    if (tensor.data_ptr() != output.data_ptr()) {
      with_device device_context(device);
      output.copy_(tensor);
    }
  } else {
    bf_output = std::make_shared<TorchTensor>(output);
  }

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorBroadcast(
      bf_tensor, bf_output, root_rank, GetOpName("broadcast", name, handle),
      device, [handle](const Status& status) {
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAllgather(::torch::Tensor tensor, ::torch::Tensor output, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  // The real output space of allgather is allocated later because we don't know the size
  // of output in advance.
  auto bf_context = std::make_shared<TorchOpContext>(device, output);
  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllgather(
      bf_tensor, bf_context, GetOpName("allgather", name, handle), device,
      [handle](const Status& status) {
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoNeighborAllgather(::torch::Tensor tensor, ::torch::Tensor output,
                        const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  // The real output space of neighbor allgather is allocated later because
  // we don't know the size of output in advance.
  auto bf_context = std::make_shared<TorchOpContext>(device, output);
  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorNeighborAllgather(
      bf_tensor, bf_context, GetOpName("neighbor.allgather", name, handle), device,
      [handle](const Status& status) {
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoNeighborAllreduce(::torch::Tensor tensor, ::torch::Tensor output,
                        int average, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  auto bf_context = std::make_shared<TorchOpContext>(device, output);
  auto bf_output = std::make_shared<TorchTensor>(tensor);

  auto enqueue_result = EnqueueTensorNeighborAllreduce(
      bf_context, bf_tensor, bf_output,
      GetOpName("neighbor.allreduce", name, handle), device,
      [handle, average, tensor, output](const Status& status) mutable {
        int first_dim = output.size(0) / bluefog_neighbor_size();
        std::vector<int64_t> shape_vector;
        shape_vector.push_back(first_dim);
        for (int idx = 1; idx < tensor.dim(); ++idx) {
          shape_vector.push_back(tensor.size(idx));
        }

        auto output_reduced = output.slice(0, 0, first_dim);
        for (int i=1; i < bluefog_neighbor_size(); i++) {
          output_reduced.add_(output.slice(0, i*first_dim, (i+1)*first_dim));
        }
        output.resize_(shape_vector);
        // Include self data as well.
        output.add_(tensor);
        // Will execute in the `device` context.
        if (average) {
          output.div_(bluefog_neighbor_size() + 1);
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int PollHandle(int handle) { return handle_manager.PollHandle(handle) ? 1 : 0; }

void WaitAndClear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

void Barrier() {
  ThrowIfError(common::CheckInitialized());
  auto handle = handle_manager.AllocateHandle();

  auto status = common::Barrier([handle](const Status& status) {
    handle_manager.MarkDone(handle, status);
  });
  ThrowIfError(status);
  // Wait until the barrier is done.
  WaitAndClear(handle);
}

// Forward declare function to add all functions in mpi_win_ops into mpi_lib module.
void AddWinOpsIntoPybind(py::module &);

PYBIND11_MODULE(mpi_lib, m) {
  // allreduce
  m.def("bluefog_torch_allreduce_async_torch_IntTensor", &DoAllreduce);
  m.def("bluefog_torch_allreduce_async_torch_LongTensor", &DoAllreduce);
  m.def("bluefog_torch_allreduce_async_torch_FloatTensor", &DoAllreduce);
  m.def("bluefog_torch_allreduce_async_torch_DoubleTensor", &DoAllreduce);
#if HAVE_CUDA
  m.def("bluefog_torch_allreduce_async_torch_cuda_IntTensor", &DoAllreduce);
  m.def("bluefog_torch_allreduce_async_torch_cuda_LongTensor", &DoAllreduce);
  m.def("bluefog_torch_allreduce_async_torch_cuda_FloatTensor", &DoAllreduce);
  m.def("bluefog_torch_allreduce_async_torch_cuda_DoubleTensor", &DoAllreduce);
#endif

  // broadcast
  m.def("bluefog_torch_broadcast_async_torch_ByteTensor", &DoBroadcast);
  m.def("bluefog_torch_broadcast_async_torch_CharTensor", &DoBroadcast);
  m.def("bluefog_torch_broadcast_async_torch_ShortTensor", &DoBroadcast);
  m.def("bluefog_torch_broadcast_async_torch_IntTensor", &DoBroadcast);
  m.def("bluefog_torch_broadcast_async_torch_LongTensor", &DoBroadcast);
  m.def("bluefog_torch_broadcast_async_torch_FloatTensor", &DoBroadcast);
  m.def("bluefog_torch_broadcast_async_torch_DoubleTensor", &DoBroadcast);
#if HAVE_CUDA
  m.def("bluefog_torch_broadcast_async_torch_cuda_IntTensor", &DoBroadcast);
  m.def("bluefog_torch_broadcast_async_torch_cuda_LongTensor", &DoBroadcast);
  m.def("bluefog_torch_broadcast_async_torch_cuda_FloatTensor", &DoBroadcast);
  m.def("bluefog_torch_broadcast_async_torch_cuda_DoubleTensor", &DoBroadcast);
#endif

  // allgather
  m.def("bluefog_torch_allgather_async_torch_ByteTensor", &DoAllgather);
  m.def("bluefog_torch_allgather_async_torch_CharTensor", &DoAllgather);
  m.def("bluefog_torch_allgather_async_torch_ShortTensor", &DoAllgather);
  m.def("bluefog_torch_allgather_async_torch_IntTensor", &DoAllgather);
  m.def("bluefog_torch_allgather_async_torch_LongTensor", &DoAllgather);
  m.def("bluefog_torch_allgather_async_torch_FloatTensor", &DoAllgather);
  m.def("bluefog_torch_allgather_async_torch_DoubleTensor", &DoAllgather);
#if HAVE_CUDA
  m.def("bluefog_torch_allgather_async_torch_cuda_IntTensor", &DoAllgather);
  m.def("bluefog_torch_allgather_async_torch_cuda_LongTensor", &DoAllgather);
  m.def("bluefog_torch_allgather_async_torch_cuda_FloatTensor", &DoAllgather);
  m.def("bluefog_torch_allgather_async_torch_cuda_DoubleTensor", &DoAllgather);
#endif

  // neighbor_allgather
  m.def("bluefog_torch_neighbor_allgather_async_torch_ByteTensor",
        &DoNeighborAllgather);
  m.def("bluefog_torch_neighbor_allgather_async_torch_CharTensor",
        &DoNeighborAllgather);
  m.def("bluefog_torch_neighbor_allgather_async_torch_ShortTensor",
        &DoNeighborAllgather);
  m.def("bluefog_torch_neighbor_allgather_async_torch_IntTensor",
        &DoNeighborAllgather);
  m.def("bluefog_torch_neighbor_allgather_async_torch_LongTensor",
        &DoNeighborAllgather);
  m.def("bluefog_torch_neighbor_allgather_async_torch_FloatTensor",
        &DoNeighborAllgather);
  m.def("bluefog_torch_neighbor_allgather_async_torch_DoubleTensor",
        &DoNeighborAllgather);
#if HAVE_CUDA
  m.def("bluefog_torch_neighbor_allgather_async_torch_cuda_IntTensor",
        &DoNeighborAllgather);
  m.def("bluefog_torch_neighbor_allgather_async_torch_cuda_LongTensor",
        &DoNeighborAllgather);
  m.def("bluefog_torch_neighbor_allgather_async_torch_cuda_FloatTensor",
        &DoNeighborAllgather);
  m.def("bluefog_torch_neighbor_allgather_async_torch_cuda_DoubleTensor",
        &DoNeighborAllgather);
#endif

  // neighbor_allreduce
  m.def("bluefog_torch_neighbor_allreduce_async_torch_IntTensor",
        &DoNeighborAllreduce);
  m.def("bluefog_torch_neighbor_allreduce_async_torch_LongTensor",
        &DoNeighborAllreduce);
  m.def("bluefog_torch_neighbor_allreduce_async_torch_FloatTensor",
        &DoNeighborAllreduce);
  m.def("bluefog_torch_neighbor_allreduce_async_torch_DoubleTensor",
        &DoNeighborAllreduce);
#if HAVE_CUDA
  m.def("bluefog_torch_neighbor_allreduce_async_torch_cuda_IntTensor",
        &DoNeighborAllreduce);
  m.def("bluefog_torch_neighbor_allreduce_async_torch_cuda_LongTensor",
        &DoNeighborAllreduce);
  m.def("bluefog_torch_neighbor_allreduce_async_torch_cuda_FloatTensor",
        &DoNeighborAllreduce);
  m.def("bluefog_torch_neighbor_allreduce_async_torch_cuda_DoubleTensor",
        &DoNeighborAllreduce);
#endif

  // basics
  m.def("bluefog_torch_poll", &PollHandle);
  m.def("bluefog_torch_wait_and_clear", &WaitAndClear);
  m.def("bluefog_torch_barrier", &Barrier);

  // one-sided communication
  AddWinOpsIntoPybind(m);
}

}  // namespace torch
}  // namespace bluefog