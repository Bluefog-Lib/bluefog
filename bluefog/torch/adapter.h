#ifndef BLUEFOG_TORCH_ADAPTER_H
#define BLUEFOG_TORCH_ADAPTER_H

#include <torch/extension.h>
#include <torch/torch.h>

#include "../common/common.h"

namespace bluefog {
namespace torch {

using namespace bluefog::common;

class TorchTensor : public Tensor {
 public:
  TorchTensor(::torch::Tensor tensor);
  virtual const DataType dtype() const override;
  virtual const TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

  // TODO(ybc) Figure out a better encapsulated way to do it.
  ::torch::Tensor MakeCopy(int device);
  ::torch::Tensor GetUnderlyingTensor();

 protected:
  ::torch::Tensor tensor_;
};

::torch::ScalarType GetTorchDataType(DataType dtype);

class TorchPersistentBuffer : public PersistentBuffer {
public:
  TorchPersistentBuffer(int device, int64_t size);
  virtual const void*
  AccessData(std::shared_ptr<OpContext> context) const override;

private:
  int device_ = CPU_DEVICE_ID;
  ::torch::Tensor tensor_;
};

class TorchOpContext : public OpContext {
public:
  TorchOpContext(int device, ::torch::Tensor output);
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) override;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) override;
  virtual Status AllocateZeros(int64_t num_elements, DataType dtype,
                                std::shared_ptr<Tensor>* tensor) override;
  virtual Framework framework() const override;

private:
  int device_ = CPU_DEVICE_ID;
  ::torch::Tensor output_;
};


void ThrowIfError(Status status);

}  // namespace torch
}  // namespace bluefog

#endif  // BLUEFOG_TORCH_ADAPTER_H