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

#ifndef BLUEFOG_TENSORFLOW_ADAPTER_H
#define BLUEFOG_TENSORFLOW_ADAPTER_H

#include <memory>

#include "../common/common.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#if HAVE_CUDA
#include "tensorflow/stream_executor/stream.h"
#endif

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
  virtual std::shared_ptr<common::Tensor> data_weight(float weight) override;
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