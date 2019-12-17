#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "adapter.h"
#include "../common/operations.h"

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
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    auto bf_context = std::make_shared<TFOpContext>(context);
    auto bf_tensor = std::make_shared<TFTensor>(tensor);
    auto bf_output = std::make_shared<TFTensor>(*output);
    auto enqueue_result = common::EnqueueTensorAllreduce(
        bf_tensor, bf_output, node_name, device,
        [context, done](const common::Status& status) {
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
};

REGISTER_KERNEL_BUILDER(Name("BluefogAllreduce").Device(::tensorflow::DEVICE_CPU),
                        BluefogAllreduceOp);
#if HAVE_CUDA
REGISTER_KERNEL_BUILDER(Name("BluefogAllreduce").Device(::tensorflow::DEVICE_GPU),
                        BluefogAllreduceOp);
#endif

REGISTER_OP("BluefogAllreduce")
    .Attr("T: {int32, int64, float16, float32, float64}")
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

}  // namespace tensorflow
} // namespace bluefog