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

#ifndef BLUEFOG_TORCH_ADAPTER_H
#define BLUEFOG_TORCH_ADAPTER_H

#include <torch/extension.h>
#include <torch/torch.h>

#if HAVE_CUDA
#include "cuda_runtime.h"
#endif

#include "../common/common.h"

namespace bluefog {
namespace torch {

class TorchTensor : public common::Tensor {
 public:
  TorchTensor(::torch::Tensor tensor);
  virtual const common::DataType dtype() const override;
  virtual const common::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual std::shared_ptr<common::Tensor> data_weight(float weight) override;
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

#if HAVE_CUDA
class TorchReadyEvent : public common::ReadyEvent {
public:
  TorchReadyEvent(int device);
  ~TorchReadyEvent();
  virtual bool Ready() const override;

private:
  int device_ = CPU_DEVICE_ID;
  cudaEvent_t cuda_event_ = nullptr;
};
#endif

std::shared_ptr<common::ReadyEvent> RecordReadyEvent(int device);

void ThrowIfError(common::Status status);

}  // namespace torch
}  // namespace bluefog

#endif  // BLUEFOG_TORCH_ADAPTER_H