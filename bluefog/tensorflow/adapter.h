#ifndef BLUEFOG_TENSORFLOW_ADAPTER_H
#define BLUEFOG_TENSORFLOW_ADAPTER_H

#include <memory>

#include "../common/common.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace bluefog {
namespace tensorflow {

::tensorflow::Status ConvertStatus(const common::Status& status);
common::Status ConvertStatus(const ::tensorflow::Status& status);

class TFPersistentBuffer : public common::PersistentBuffer {
 public:
  TFPersistentBuffer(::tensorflow::OpKernelContext* context, int64_t size);
  virtual const void* AccessData(
      std::shared_ptr<common::OpContext> context) const override;

 private:
  std::shared_ptr<::tensorflow::PersistentTensor> tensor_;
};

class TFTensor : public common::Tensor {
 public:
  TFTensor(::tensorflow::Tensor& tensor);
  virtual const common::DataType dtype() const override;
  virtual const common::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

 protected:
  ::tensorflow::Tensor tensor_;
};

class TFOpContext : public common::OpContext {
 public:
  TFOpContext(::tensorflow::OpKernelContext* context);
  virtual common::Status AllocatePersistent(
      int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) override;
  virtual common::Status AllocateOutput(
      common::TensorShape shape,
      std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Status AllocateZeros(
      int64_t num_elements, common::DataType dtype,
      std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Framework framework() const override;
  ::tensorflow::OpKernelContext* GetKernelContext() const;

 private:
  ::tensorflow::OpKernelContext* context_ = nullptr;
};

}  // namespace tensorflow
}  // namespace bluefog

#endif  // BLUEFOG_TENSORFLOW_ADAPTER_H