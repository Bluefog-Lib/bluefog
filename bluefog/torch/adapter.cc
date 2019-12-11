#include "adapter.h"
#include "cuda_util.h"
#include "../common/logging.h"

namespace bluefog {
namespace torch {

using ::bluefog::common::DataType;
using ::bluefog::common::Framework;
using ::bluefog::common::Status;
using ::bluefog::common::StatusType;
using ::bluefog::common::OpContext;

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

int64_t TorchTensor::size() const {
  return tensor_.element_size() * tensor_.numel();
}

::torch::Tensor TorchTensor::MakeCopy(int device) {
  with_device device_context(device);
  ::torch::Tensor t;
  if (device == CPU_DEVICE_ID) { 
    t = ::torch::ones(tensor_.sizes(), ::torch::device(::torch::kCPU).dtype(tensor_.dtype()));
  } else {
    t = ::torch::ones(tensor_.sizes(),
                      ::torch::device(::torch::kCUDA).dtype(tensor_.dtype()));
  }
  t.copy_(tensor_, /*nonblocking=*/false);
  return t;
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
