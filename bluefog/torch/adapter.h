#ifndef BLUEFOG_TORCH_ADAPTER_H
#define BLUEFOG_TORCH_ADAPTER_H

#include <torch/extension.h>
#include <torch/torch.h>

#include "../common/common.h"

namespace bluefog {
namespace torch {

class TorchTensor : public common::Tensor {
 public:
  TorchTensor(::torch::Tensor tensor);
  virtual const common::DataType dtype() const override;
  virtual const common::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

  // TODO(ybc) Figure out a better encapsulated way to do it.
  ::torch::Tensor MakeCopy(int device);
  ::torch::Tensor GetUnderlyingTensor();

 protected:
  ::torch::Tensor tensor_;
};

::torch::ScalarType GetTorchDataType(common::DataType dtype);

class TorchPersistentBuffer : public common::PersistentBuffer {
 public:
  TorchPersistentBuffer(int device, int64_t size);
  virtual const void* AccessData(
      std::shared_ptr<common::OpContext> context) const override;

 private:
  int device_ = CPU_DEVICE_ID;
  ::torch::Tensor tensor_;
};

class TorchOpContext : public common::OpContext {
 public:
  TorchOpContext(int device, ::torch::Tensor output);
  virtual common::Status AllocatePersistent(
      int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) override;
  virtual common::Status AllocateOutput(
      common::TensorShape shape,
      std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Status AllocateZeros(
      int64_t num_elements, common::DataType dtype,
      std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Framework framework() const override;

 private:
  int device_ = CPU_DEVICE_ID;
  ::torch::Tensor output_;
};

void ThrowIfError(common::Status status);

}  // namespace torch
}  // namespace bluefog

#endif  // BLUEFOG_TORCH_ADAPTER_H