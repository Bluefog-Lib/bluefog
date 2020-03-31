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

#include "adapter.h"

#include "../common/logging.h"

namespace bluefog {
namespace tensorflow {

common::Status ConvertStatus(const ::tensorflow::Status& status) {
  switch (status.code()) {
    case ::tensorflow::error::Code::OK:
      return common::Status::OK();
    case ::tensorflow::error::Code::UNKNOWN:
      return common::Status::UnknownError(status.error_message());
    case ::tensorflow::error::Code::FAILED_PRECONDITION:
      return common::Status::PreconditionError(status.error_message());
    case ::tensorflow::error::Code::ABORTED:
      return common::Status::Aborted(status.error_message());
    case ::tensorflow::error::Code::INVALID_ARGUMENT:
      return common::Status::InvalidArgument(status.error_message());
    default:
      return common::Status::UnknownError("Unknown error.");
  }
}

::tensorflow::Status ConvertStatus(const common::Status& status) {
  switch (status.type()) {
    case common::StatusType::OK:
      return ::tensorflow::Status::OK();
    case common::StatusType::UNKNOWN_ERROR:
      return ::tensorflow::errors::Unknown(status.reason());
    case common::StatusType::PRECONDITION_ERROR:
      return ::tensorflow::errors::FailedPrecondition(status.reason());
    case common::StatusType::ABORTED:
      return ::tensorflow::errors::Aborted(status.reason());
    case common::StatusType::INVALID_ARGUMENT:
      return ::tensorflow::errors::InvalidArgument(status.reason());
    default:
      return ::tensorflow::errors::Unknown("Unknown error.");
  }
}

TFPersistentBuffer::TFPersistentBuffer(::tensorflow::OpKernelContext* context,
                                       int64_t size) {
  tensor_ = std::make_shared<::tensorflow::PersistentTensor>();
  ::tensorflow::TensorShape buffer_shape;
  buffer_shape.AddDim(size);
  ::tensorflow::Tensor* unused;
  ::tensorflow::Status status = context->allocate_persistent(
      ::tensorflow::DT_INT8, buffer_shape, tensor_.get(), &unused);
  if (!status.ok()) {
    throw status;
  }
#if HAVE_CUDA
  // On GPU allocation is asynchronous, we need to wait for it to
  // complete.
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    device_context->stream()->BlockHostUntilDone();
  }
#endif
}

const void* TFPersistentBuffer::AccessData(
    std::shared_ptr<common::OpContext> context) const {
  // It's safe to cast context to TFOpContext, since only TFOpContext creates
  // TFPersistentBuffer.
  return (const void*)tensor_
      ->AccessTensor(
          std::dynamic_pointer_cast<TFOpContext>(context)->GetKernelContext())
      ->tensor_data()
      .data();
}

TFTensor::TFTensor(::tensorflow::Tensor& tensor) : tensor_(tensor) {}

const common::DataType TFTensor::dtype() const {
  switch (tensor_.dtype()) {
    case ::tensorflow::DT_UINT8:
      return common::DataType::BLUEFOG_UINT8;
    case ::tensorflow::DT_INT8:
      return common::DataType::BLUEFOG_INT8;
    case ::tensorflow::DT_UINT16:
      return common::DataType::BLUEFOG_UINT16;
    case ::tensorflow::DT_INT16:
      return common::DataType::BLUEFOG_INT16;
    case ::tensorflow::DT_INT32:
      return common::DataType::BLUEFOG_INT32;
    case ::tensorflow::DT_INT64:
      return common::DataType::BLUEFOG_INT64;
    case ::tensorflow::DT_HALF:
      return common::DataType::BLUEFOG_FLOAT16;
    case ::tensorflow::DT_FLOAT:
      return common::DataType::BLUEFOG_FLOAT32;
    case ::tensorflow::DT_DOUBLE:
      return common::DataType::BLUEFOG_FLOAT64;
    case ::tensorflow::DT_BOOL:
      return common::DataType::BLUEFOG_BOOL;
    default:
      throw std::logic_error("Invalid tensor type.");
  }
}

const common::TensorShape TFTensor::shape() const {
  common::TensorShape shape;
  for (auto dim : tensor_.shape()) {
    shape.AddDim(dim.size);
  }
  return shape;
}

const void* TFTensor::data() const {
  return (const void*)tensor_.tensor_data().data();
}

std::shared_ptr<common::Tensor> TFTensor::data_weight(float weight) {
  throw std::runtime_error("Tensorflow with weight is not implemented yet.");
};

int64_t TFTensor::size() const { return (int64_t)tensor_.tensor_data().size(); }

TFOpContext::TFOpContext(::tensorflow::OpKernelContext* context)
    : context_(context) {}

common::Status TFOpContext::AllocatePersistent(
    int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) {
  try {
    *tensor = std::make_shared<TFPersistentBuffer>(context_, size);
    return common::Status::OK();
  } catch (::tensorflow::Status& status) {
    return ConvertStatus(status);
  }
}

common::Status TFOpContext::AllocateOutput(
    common::TensorShape shape, std::shared_ptr<common::Tensor>* tensor) {
  ::tensorflow::TensorShape tf_shape;
  for (int idx = 0; idx < shape.dims(); ++idx) {
    tf_shape.AddDim(shape.dim_size(idx));
  }
  ::tensorflow::Tensor* tf_tensor;
  ::tensorflow::Status status =
      context_->allocate_output(0, tf_shape, &tf_tensor);
  if (status.ok()) {
    *tensor = std::make_shared<TFTensor>(*tf_tensor);
  }
#if HAVE_CUDA
  // On GPU allocation is asynchronous, we need to wait for it to
  // complete.
  auto device_context = context_->op_device_context();
  if (device_context != nullptr) {
    device_context->stream()->BlockHostUntilDone();
  }
#endif
  return ConvertStatus(status);
}

common::Status TFOpContext::AllocateZeros(
    int64_t num_elements, common::DataType dtype,
    std::shared_ptr<common::Tensor>* tensor) {
  return common::Status::PreconditionError(
      "AllocateZeros is not supported for TensorFlow yet.");
}

common::Framework TFOpContext::framework() const {
  return common::Framework::TENSORFLOW;
}

::tensorflow::OpKernelContext* TFOpContext::GetKernelContext() const {
  return context_;
}

}  // namespace tensorflow
}  // namespace bluefog
