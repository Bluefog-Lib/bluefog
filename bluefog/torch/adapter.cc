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

#if HAVE_CUDA
#include <THC/THC.h>
#endif

#include "adapter.h"
#include "../common/cuda_util.h"
#include "../common/logging.h"

#if HAVE_CUDA
extern THCState* state;
#endif

namespace bluefog {
namespace torch {

using ::bluefog::common::DataType;
using ::bluefog::common::Framework;
using ::bluefog::common::Status;
using ::bluefog::common::StatusType;
using ::bluefog::common::OpContext;
using ::bluefog::common::with_device;

TorchTensor::TorchTensor(::torch::Tensor tensor) : tensor_(tensor) {}

const DataType TorchTensor::dtype() const {
  switch (tensor_.scalar_type()) {
    case ::torch::kByte:
      return DataType::BLUEFOG_UINT8;
    case ::torch::kChar:
      return DataType::BLUEFOG_INT8;
    case ::torch::kShort:
      return DataType::BLUEFOG_INT16;
    case ::torch::kInt:
      return DataType::BLUEFOG_INT32;
    case ::torch::kLong:
      return DataType::BLUEFOG_INT64;
    case ::torch::kHalf:
      return DataType::BLUEFOG_FLOAT16;
    case ::torch::kFloat:
      return DataType::BLUEFOG_FLOAT32;
    case ::torch::kDouble:
      return DataType::BLUEFOG_FLOAT64;
    default:
      throw std::logic_error("Invalid or unsupported tensor type.");
  }
}

const common::TensorShape TorchTensor::shape() const {
  common::TensorShape shape;
  for (int idx = 0; idx < tensor_.dim(); ++idx) {
    shape.AddDim(tensor_.size(idx));
  }
  return shape;
}

const void* TorchTensor::data() const { return tensor_.data_ptr(); }

std::shared_ptr<common::Tensor> TorchTensor::data_weight(float weight) {
  if (weight == 1.0) {
    return std::make_shared<TorchTensor>(tensor_);
  } else {
    // Note we call mul instead of mul_
    auto t = tensor_.mul(weight);
    return std::make_shared<TorchTensor>(t);
  }
}

int64_t TorchTensor::size() const {
  return tensor_.element_size() * tensor_.numel();
}

::torch::Tensor TorchTensor::MakeCopy(int device) {
  with_device device_context(device);
  return tensor_.clone();
}

::torch::Tensor TorchTensor::GetUnderlyingTensor() { return tensor_; }

::torch::ScalarType GetTorchDataType(DataType dtype) {
  switch (dtype) {
    case DataType::BLUEFOG_UINT8:
      return ::torch::kByte;
    case DataType::BLUEFOG_INT8:
      return ::torch::kChar;
    case DataType::BLUEFOG_INT16:
      return ::torch::kShort;
    case DataType::BLUEFOG_INT32:
      return ::torch::kInt;
    case DataType::BLUEFOG_INT64:
      return ::torch::kLong;
    case DataType::BLUEFOG_FLOAT16:
      return ::torch::kHalf;
    case DataType::BLUEFOG_FLOAT32:
      return ::torch::kFloat;
    case DataType::BLUEFOG_FLOAT64:
      return ::torch::kDouble;
    default:
      throw std::logic_error("Invalid data type.");
  }
}

TorchPersistentBuffer::TorchPersistentBuffer(int device, int64_t size)
    : device_(device) {
  with_device device_context(device_);
  if (device_ == CPU_DEVICE_ID) {
    tensor_ = ::torch::empty(size, ::torch::device(::torch::kCPU).dtype(::torch::kByte));
  } else {
    tensor_ = ::torch::empty(size, ::torch::device(::torch::kCUDA).dtype(::torch::kByte));
  }
}

const void* TorchPersistentBuffer::AccessData(
    std::shared_ptr<OpContext> context) const {
  return tensor_.data_ptr();
}

TorchOpContext::TorchOpContext(int device, ::torch::Tensor output)
    : device_(device), output_(output) {}

Status
TorchOpContext::AllocatePersistent(int64_t size,
                                   std::shared_ptr<common::PersistentBuffer>* tensor) {
  // Allocation errors are handled using PyTorch exceptions.
  *tensor = std::make_shared<TorchPersistentBuffer>(device_, size);
  return Status::OK();
}

Status TorchOpContext::AllocateOutput(common::TensorShape shape,
                                      std::shared_ptr<common::Tensor>* tensor) {
  std::vector<int64_t> shape_vector;
  shape_vector.reserve(shape.dims());
  for (int idx = 0; idx < shape.dims(); ++idx) {
    shape_vector.push_back(shape.dim_size(idx));
  }
  with_device device_context(device_);
  output_.resize_(shape_vector);
  BFLOG(TRACE) << "Output tensor after allocated " << output_.scalar_type()
               << " " << output_.size(0) << " " << output_.device();
  *tensor = std::make_shared<TorchTensor>(output_);
  return Status::OK();
}

Status TorchOpContext::AllocateZeros(int64_t num_elements, DataType dtype,
                                     std::shared_ptr<common::Tensor>* tensor) {
  with_device device_context(device_);
  auto torch_data_type = GetTorchDataType(dtype);
  ::torch::DeviceType device_type =
      device_ != CPU_DEVICE_ID ? ::torch::kCUDA : ::torch::kCPU;
  ::torch::Tensor zero_tensor = ::torch::zeros(
      num_elements, ::torch::device(device_type).dtype(torch_data_type));
  *tensor = std::make_shared<TorchTensor>(zero_tensor);
  return Status::OK();
}

Framework TorchOpContext::framework() const {
  return Framework::PYTORCH;
}

#if HAVE_CUDA
struct ReadyEventRegistry {
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex mutex;
};

static ReadyEventRegistry ready_event_registry;

TorchReadyEvent::TorchReadyEvent(int device) : device_(device) {
  assert(device_ != CPU_DEVICE_ID);

  with_device device_context(device_);
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    if (!queue.empty()) {
      cuda_event_ = queue.front();
      queue.pop();
    } else {
      THCudaCheck(cudaEventCreateWithFlags(
          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
    }
  }
  auto stream = THCState_getCurrentStreamOnDevice(state, device_);
  THCudaCheck(cudaEventRecord(cuda_event_, stream));
}

TorchReadyEvent::~TorchReadyEvent() {
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    queue.push(cuda_event_);
  }
}

bool TorchReadyEvent::Ready() const {
  auto status = cudaEventQuery(cuda_event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  THCudaCheck(status);
  return true;
}
#endif

// On GPU this event will signal that GPU computations are done and data is
// ready.
std::shared_ptr<common::ReadyEvent> RecordReadyEvent(int device) {
  if (device == CPU_DEVICE_ID) {
    return std::shared_ptr<common::ReadyEvent>();
  } else {
#if HAVE_CUDA
    return std::make_shared<TorchReadyEvent>(device);
#else
    throw std::logic_error("Internal error. Requested ReadyEvent "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

void ThrowIfError(Status status) {
  switch (status.type()) {
    case StatusType::OK:
      return;
    case StatusType::PRECONDITION_ERROR:
      throw std::logic_error(status.reason());
    case StatusType::ABORTED:
      throw std::runtime_error(status.reason());
    case StatusType::INVALID_ARGUMENT:
      throw std::invalid_argument(status.reason());
    default:  // Includes UNKNOWN_ERROR
      throw std::runtime_error(status.reason());
  }
}

}  // namespace torch
}  // namespace bluefog
