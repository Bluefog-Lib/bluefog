#include "adapter.h"

#include "../common/logging.h"

namespace bluefog {
namespace tensorflow {

common::Status ConvertStatus(const common::Status& status) {
  switch (status.type()) {
    case common::StatusType::OK:
      return common::Status::OK();
    case common::StatusType::UNKNOWN_ERROR:
      return common::Status::UnknownError(status.reason());
    case common::StatusType::PRECONDITION_ERROR:
      return common::Status::PreconditionError(status.reason());
    case common::StatusType::ABORTED:
      return common::Status::Aborted(status.reason());
    case common::StatusType::INVALID_ARGUMENT:
      return common::Status::InvalidArgument(status.reason());
    default:
      return common::Status::UnknownError("Unknown error.");
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

int64_t TFTensor::size() const { return (int64_t)tensor_.tensor_data().size(); }

TFOpContext::TFOpContext(::tensorflow::OpKernelContext* context)
    : context_(context) {}

common::Status TFOpContext::AllocatePersistent(
    int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) {
  try {
    *tensor = std::make_shared<TFPersistentBuffer>(context_, size);
    return common::Status::OK();
  } catch (common::Status& status) {
    return ConvertStatus(status);
  }
}

}  // namespace tensorflow
}  // namespace bluefog
