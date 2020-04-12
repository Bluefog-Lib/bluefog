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

#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>

#include "adapter.h"
#include "handle_manager.h"
#include "../common/cuda_util.h"
#include "../common/logging.h"
#include "../common/operations.h"
#include "../common/timeline.h"

namespace bluefog {
namespace torch {

using ::bluefog::common::bluefog_load_topology;
using ::bluefog::common::bluefog_load_topology_weights;
using ::bluefog::common::bluefog_neighbor_size;
using ::bluefog::common::bluefog_rank;
using ::bluefog::common::bluefog_size;
using ::bluefog::common::with_device;
using ::bluefog::common::GetBluefogTimeline;
using ::bluefog::common::Status;
using ::bluefog::common::Timeline;

// static here means Local/private variable.
static HandleManager handle_manager;

static const char* BLUEFOG_OPS_ON_CPU = std::getenv("BLUEFOG_OPS_ON_CPU");
static const bool OPS_ON_CPU =
    (BLUEFOG_OPS_ON_CPU != nullptr) && (*BLUEFOG_OPS_ON_CPU == '1');

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
  auto op_name = GetOpName("allreduce", name, handle);

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);
  timeline_ptr->ActivityStart(op_name, "ENQUEUE_ALLREDUCE");

  // Note callback function will be called by different thread.
  std::thread::id tid = std::this_thread::get_id();

  if (OPS_ON_CPU && tensor.device().is_cuda()) {
    ::torch::Tensor cpu_buffer =
        tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false);
    // When input and out are the same, mpi_allreduce use IN_PLACE mode.
    // Because we will copy from cpu to gpu anway, there is no reason
    // allocate two cpu memories.
    auto bf_tensor = std::make_shared<TorchTensor>(cpu_buffer);
    auto bf_output = bf_tensor;

    auto enqueue_result = EnqueueTensorAllreduce(
        bf_tensor, bf_tensor, op_name, device,
        [handle, average, output, cpu_buffer, device, op_name, tid,
         timeline_ptr](const Status& status) mutable {
          with_device device_guard(device);
          output.copy_(cpu_buffer);

          // Will execute in the `device` context.
          if (average && bluefog_size() > 1) {
            output.div_(bluefog_size());
          }
          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name, &tid);  // ENQUEUE
        });
    ThrowIfError(enqueue_result);
  } else {
    auto bf_tensor = std::make_shared<TorchTensor>(tensor);
    auto bf_output = std::make_shared<TorchTensor>(output);

    auto enqueue_result = EnqueueTensorAllreduce(
        bf_tensor, bf_output, op_name, device,
        [handle, average, output, op_name, tid, timeline_ptr](const Status& status) mutable {
          // Will execute in the `device` context.
          if (average && bluefog_size() > 1) {
            output.div_(bluefog_size());
          }
          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name, &tid);  // ENQUEUE
        });
    ThrowIfError(enqueue_result);
  }
  return handle;
}

int DoBroadcast(::torch::Tensor tensor, ::torch::Tensor output, int root_rank,
                const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto op_name = GetOpName("broadcast", name, handle);

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);
  timeline_ptr->ActivityStart(op_name, "ENQUEUE_BROADCAST");
  // Note callback function will be called by different thread.
  std::thread::id tid = std::this_thread::get_id();

  auto bf_tensor = std::make_shared<TorchTensor>(tensor);
  std::shared_ptr<common::Tensor> bf_output = nullptr;

  if (OPS_ON_CPU && tensor.device().is_cuda()) {
    ::torch::Tensor cpu_buffer =
        tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false);
    auto bf_tensor = std::make_shared<TorchTensor>(cpu_buffer);

    auto enqueue_result = EnqueueTensorBroadcast(
        bf_tensor, bf_tensor, root_rank, op_name, device,
        [handle, output, cpu_buffer, device, op_name, tid,
         timeline_ptr](const Status& status) mutable {
          with_device device_guard(device);
          output.copy_(cpu_buffer);
          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name, &tid);  // ENQUEUE
        });
    ThrowIfError(enqueue_result);
  } else {
    if (bluefog_rank() == root_rank) {
      if (tensor.data_ptr() != output.data_ptr()) {
        with_device device_context(device);
        output.copy_(tensor);
      }
    } else {
      bf_output = std::make_shared<TorchTensor>(output);
    }

    auto enqueue_result = EnqueueTensorBroadcast(
        bf_tensor, bf_output, root_rank, op_name, device,
        [handle, op_name, tid, timeline_ptr](const Status& status) {
          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name, &tid);  // ENQUEUE
        });
    ThrowIfError(enqueue_result);
  }
  return handle;
}

int DoAllgather(::torch::Tensor tensor, ::torch::Tensor output, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto op_name = GetOpName("allgather", name, handle);

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);
  timeline_ptr->ActivityStart(op_name, "ENQUEUE_ALLGATHER");
  // Note callback function will be called by different thread.
  std::thread::id tid = std::this_thread::get_id();

  if (OPS_ON_CPU && tensor.device().is_cuda()) {
    ::torch::Tensor cpu_buffer =
        tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false);
    auto bf_tensor = std::make_shared<TorchTensor>(cpu_buffer);
    auto cpu_output = ::torch::empty_like(cpu_buffer);
    auto bf_context =
        std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_output);

    auto enqueue_result = EnqueueTensorAllgather(
        bf_tensor, bf_context, op_name, device,
        [handle, cpu_output, device, output, op_name, tid,
         timeline_ptr](const Status& status) mutable {
          with_device device_guard(device);
          // output needs to be resized before copying in the CPU tensor.
          output.resize_(cpu_output.sizes());
          output.copy_(cpu_output);
          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name, &tid);  // ENQUEUE
        });
    ThrowIfError(enqueue_result);
  } else {
    auto bf_tensor = std::make_shared<TorchTensor>(tensor);
    // The real output space of allgather is allocated later because we don't
    // know the size of output in advance.
    auto bf_context = std::make_shared<TorchOpContext>(device, output);
    auto enqueue_result = EnqueueTensorAllgather(
        bf_tensor, bf_context, op_name, device,
        [handle, op_name, tid, timeline_ptr](const Status& status) {
          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name, &tid);  // ENQUEUE
        });
    ThrowIfError(enqueue_result);
  }
  return handle;
}

int DoNeighborAllgather(::torch::Tensor tensor, ::torch::Tensor output,
                        const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto op_name = GetOpName("neighbor.allgather", name, handle);

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);
  timeline_ptr->ActivityStart(op_name, "ENQUEUE_NEIGHBOR_ALLGATHER");
  // Note callback function will be called by different thread.
  std::thread::id tid = std::this_thread::get_id();

  if (OPS_ON_CPU && tensor.device().is_cuda()) {
    ::torch::Tensor cpu_buffer =
        tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false);
    auto bf_tensor = std::make_shared<TorchTensor>(cpu_buffer);
    auto cpu_output = ::torch::empty_like(cpu_buffer);
    auto bf_context =
        std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_output);

    auto enqueue_result = EnqueueTensorNeighborAllgather(
        bf_tensor, bf_context, op_name, device,
        [handle, cpu_output, device, output, op_name, tid,
         timeline_ptr](const Status& status) mutable {
          with_device device_guard(device);
          // output needs to be resized before copying in the CPU tensor.
          output.resize_(cpu_output.sizes());
          output.copy_(cpu_output);
          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name);  // ENQUEUE
        });
    ThrowIfError(enqueue_result);
  } else {
    auto bf_tensor = std::make_shared<TorchTensor>(tensor);
    // The real output space of neighbor allgather is allocated later because
    // we don't know the size of output in advance.
    auto bf_context = std::make_shared<TorchOpContext>(device, output);
    auto enqueue_result = EnqueueTensorNeighborAllgather(
        bf_tensor, bf_context, op_name, device,
        [handle, op_name, tid, timeline_ptr](const Status& status) {
          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name);  // ENQUEUE
        });
    ThrowIfError(enqueue_result);
  }
  return handle;
}

int DoNeighborAllreduce(::torch::Tensor tensor, ::torch::Tensor output,
                        int average, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto op_name = GetOpName("neighbor.allreduce", name, handle);

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);
  timeline_ptr->ActivityStart(op_name, "ENQUEUE_NEIGHBOR_ALLREDUCE");
  // Note callback function will be called by different thread.
  std::thread::id tid = std::this_thread::get_id();

  if (OPS_ON_CPU && tensor.device().is_cuda()) {
    ::torch::Tensor cpu_buffer =
        tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/false);
    auto bf_tensor = std::make_shared<TorchTensor>(cpu_buffer);
    auto cpu_output = ::torch::empty_like(cpu_buffer);
    auto bf_context =
        std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_output);
    auto bf_output = std::make_shared<TorchTensor>(cpu_output);
    auto enqueue_result = EnqueueTensorNeighborAllreduce(
        bf_context, bf_tensor, bf_output, op_name, device,
        [handle, average, cpu_output, tensor, output, device, op_name, tid,
         timeline_ptr](const Status& status) mutable {
          with_device device_guard(device);
          output.resize_(cpu_output.sizes());
          output.copy_(cpu_output);
          
          int first_dim = output.size(0) / bluefog_neighbor_size();
          std::vector<int64_t> shape_vector;
          shape_vector.push_back(first_dim);
          for (int idx = 1; idx < tensor.dim(); ++idx) {
            shape_vector.push_back(tensor.size(idx));
          }

          // 1) For a distributed graph topology, created with
          // MPI_Dist_graph_create, the sequence of neighbors in the send and
          // receive buffers at each process is defined as the sequence returned
          // by MPI_Dist_graph_neighbors for destinations and sources,
          // respectively. 2) MPI_Dist_graph_neighbors: If the communicator was
          // created with MPI_Dist_graph_create_adjacent then the order of the
          // values in sources and destinations is identical to the input that
          // was used by the process with the same rank in comm_old in the
          // creation call.
          int indgree = 0;
          int outdegree = 0;
          int* sources_ptr = nullptr;
          int* destinations_ptr = nullptr;
          bluefog_load_topology(&indgree, sources_ptr, &outdegree,
                                destinations_ptr);

          float self_weight = 1.0;
          const std::unordered_map<int, float>* neighbor_weights_ptr;
          int is_weighted = bluefog_load_topology_weights(self_weight, neighbor_weights_ptr);
          // No matter the topology is weighted or not, if we do not need
          // average for the neighbor allreduce result, the weights will not be
          // used.
          if (average && (is_weighted == 1)) {
            auto output_reduced = output.slice(0, 0, first_dim);
            for (int i = 0; i < indgree; i++) {
              float weight = 0.0;
              auto it = neighbor_weights_ptr->find(*(sources_ptr + i));
              if (it != neighbor_weights_ptr->end()) {
                weight = it->second;
              }

              if (i == 0) {
                output_reduced.mul_(weight);
              } else {
                output_reduced.add_(
                    output.slice(0, i * first_dim, (i + 1) * first_dim)
                        .mul_(weight));
              }
            }
            output.resize_(shape_vector);
            output.add_(tensor.mul_(self_weight));
          } else {
            auto output_reduced = output.slice(0, 0, first_dim);
            for (int i = 1; i < bluefog_neighbor_size(); i++) {
              output_reduced.add_(
                  output.slice(0, i * first_dim, (i + 1) * first_dim));
            }
            output.resize_(shape_vector);
            // Include self data as well.
            output.add_(tensor);
            if (average) {
              output.div_(bluefog_neighbor_size() + 1);
            }
          }

          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name);  // ENQUEUE
        });

    ThrowIfError(enqueue_result);
  } else {
    auto bf_tensor = std::make_shared<TorchTensor>(tensor);
    auto bf_context = std::make_shared<TorchOpContext>(device, output);
    auto bf_output = std::make_shared<TorchTensor>(tensor);

    auto enqueue_result = EnqueueTensorNeighborAllreduce(
        bf_context, bf_tensor, bf_output, op_name, device,
        [handle, average, tensor, output, op_name, tid,
         timeline_ptr](const Status& status) mutable {
          int first_dim = output.size(0) / bluefog_neighbor_size();
          std::vector<int64_t> shape_vector;
          shape_vector.push_back(first_dim);
          for (int idx = 1; idx < tensor.dim(); ++idx) {
            shape_vector.push_back(tensor.size(idx));
          }

          // 1) For a distributed graph topology, created with
          // MPI_Dist_graph_create, the sequence of neighbors in the send and
          // receive buffers at each process is defined as the sequence returned
          // by MPI_Dist_graph_neighbors for destinations and sources,
          // respectively. 2) MPI_Dist_graph_neighbors: If the communicator was
          // created with MPI_Dist_graph_create_adjacent then the order of the
          // values in sources and destinations is identical to the input that
          // was used by the process with the same rank in comm_old in the
          // creation call.
          int indgree = 0;
          int outdegree = 0;
          int* sources_ptr = nullptr;
          int* destinations_ptr = nullptr;
          bluefog_load_topology(&indgree, sources_ptr, &outdegree,
                                destinations_ptr);

          float self_weight = 1.0;
          const std::unordered_map<int, float>* neighbor_weights_ptr;
          int is_weighted = bluefog_load_topology_weights(self_weight, neighbor_weights_ptr);
          // No matter the topology is weighted or not, if we do not need
          // average for the neighbor allreduce result, the weights will not be
          // used.
          if (average && (is_weighted == 1)) {
            auto output_reduced = output.slice(0, 0, first_dim);
            for (int i = 0; i < indgree; i++) {
              float weight = 0.0;
              auto it = neighbor_weights_ptr->find(*(sources_ptr + i));
              if (it != neighbor_weights_ptr->end()) {
                weight = it->second;
              }

              if (i == 0) {
                output_reduced.mul_(weight);
              } else {
                output_reduced.add_(
                    output.slice(0, i * first_dim, (i + 1) * first_dim)
                        .mul_(weight));
              }
            }
            output.resize_(shape_vector);
            output.add_(tensor.mul_(self_weight));
          } else {
            auto output_reduced = output.slice(0, 0, first_dim);
            for (int i = 1; i < bluefog_neighbor_size(); i++) {
              output_reduced.add_(
                  output.slice(0, i * first_dim, (i + 1) * first_dim));
            }
            output.resize_(shape_vector);
            // Include self data as well.
            output.add_(tensor);
            if (average) {
              output.div_(bluefog_neighbor_size() + 1);
            }
          }

          handle_manager.MarkDone(handle, status);
          timeline_ptr->ActivityEnd(op_name);  // ENQUEUE
        });
    ThrowIfError(enqueue_result);
  }
  return handle;
}

int PollHandle(int handle) { return handle_manager.PollHandle(handle) ? 1 : 0; }

void WaitAndClear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
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
