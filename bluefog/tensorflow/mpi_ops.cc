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

#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>

#include "../common/operations.h"
#include "adapter.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace bluefog {
namespace tensorflow {

using ::tensorflow::AsyncOpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;

namespace {
int GetDeviceID(OpKernelContext* context) {
  int device = CPU_DEVICE_ID;
  if (context->device() != nullptr &&
      context->device()->tensorflow_gpu_device_info() != nullptr) {
    device = context->device()->tensorflow_gpu_device_info()->gpu_id;
  }
  return device;
}
}  // namespace

class BluefogAllreduceOp : public AsyncOpKernel {
 public:
  explicit BluefogAllreduceOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    ::tensorflow::Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, tensor.shape(), &output), done);

    auto bf_context = std::make_shared<TFOpContext>(context);
    auto bf_tensor = std::make_shared<TFTensor>(tensor);
    auto bf_output = std::make_shared<TFTensor>(*output);
    auto enqueue_result = EnqueueTensorAllreduce(
        bf_tensor, bf_output, node_name, device,
        [context, done](const common::Status& status) {
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BluefogAllreduce").Device(::tensorflow::DEVICE_CPU),
    BluefogAllreduceOp);
#if HAVE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("BluefogAllreduce").Device(::tensorflow::DEVICE_GPU),
    BluefogAllreduceOp);
#endif

REGISTER_OP("BluefogAllreduce")
    .Attr("T: {int32, int64, float32, float64}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
allreduce.

Arguments
    tensor:     A tensor to reduce.

Output
    sum:    A tensor with the same shape as `tensor`, summed across all MPI processes.
)doc");


class BluefogBroadcastOp : public AsyncOpKernel {
 public:
  explicit BluefogBroadcastOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    ::tensorflow::Tensor* output = nullptr;
    if (common::bluefog_rank() == root_rank_) {
      context->set_output(0, tensor);
    } else {
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, tensor.shape(), &output), done);
    }
    auto bf_context = std::make_shared<TFOpContext>(context);
    auto bf_tensor = std::make_shared<TFTensor>(tensor);
    std::shared_ptr<TFTensor> bf_output = nullptr;
    if (output != nullptr) {
      bf_output = std::make_shared<TFTensor>(*output);
    }
    auto enqueue_result = EnqueueTensorBroadcast(
        bf_tensor, bf_output, root_rank_, node_name, device,
        [context, done](const common::Status& status) {
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }

 private:
  int root_rank_;
};

REGISTER_KERNEL_BUILDER(
    Name("BluefogBroadcast").Device(::tensorflow::DEVICE_CPU),
    BluefogBroadcastOp);
#if HAVE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("BluefogBroadcast").Device(::tensorflow::DEVICE_GPU),
    BluefogBroadcastOp);
#endif

REGISTER_OP("BluefogBroadcast")
    .Attr("T: {int32, int64, float32, float64, bool}")
    .Attr("root_rank: int")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Broadcast on a tensor. All other processes that do a broadcast
on a tensor with the same name must have the same dimension for that tensor.

Arguments
    tensor:     A tensor to broadcast.
    root_rank:  Rank that will send data, other ranks will receive data.

Output
    output:    A tensor with the same shape as `tensor` and same value as
               `tensor` on root rank.
)doc");

class BluefogAllgatherOp : public AsyncOpKernel {
public:
  explicit BluefogAllgatherOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    // We cannot pre-allocate output for allgather, since shape of result is 
    // only known after all ranks make a request.
    auto bf_context = std::make_shared<TFOpContext>(context);
    auto bf_tensor = std::make_shared<TFTensor>(tensor);
    auto enqueue_result = EnqueueTensorAllgather(
        bf_tensor, bf_context, node_name, device,
        [context, done](const common::Status& status) {
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BluefogAllgather").Device(::tensorflow::DEVICE_CPU),
    BluefogAllgatherOp);
#if HAVE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("BluefogAllgather").Device(::tensorflow::DEVICE_GPU),
    BluefogAllgatherOp);
#endif

REGISTER_OP("BluefogAllgather")
    .Attr("T: {int32, int64, float32, float64, bool}")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allgather on a tensor. All other processes that do a gather on a
tensor with the same name must have the same rank for that tensor, and have the
same dimension on all but the first dimension.

Arguments
    tensor:     A tensor to gather.

Output
    gathered:    A tensor with the same shape as `tensor` except for the first dimension.
)doc");

}  // namespace tensorflow
}  // namespace bluefog