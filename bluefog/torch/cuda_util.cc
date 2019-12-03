#if HAVE_CUDA
#include "cuda_runtime.h"
#include <THC/THC.h>
#endif

#include "../common/common.h"
#include "cuda_util.h"

namespace bluefog {
namespace torch {

with_device::with_device(int device) {
  if (device == CPU_DEVICE_ID) {
    restore_device_ = CPU_DEVICE_ID;
  } else {
#if HAVE_CUDA
    THCudaCheck(cudaGetDevice(&restore_device_));
    THCudaCheck(cudaSetDevice(device));
#else
    throw std::logic_error("Internal error. Requested device context manager "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

with_device::~with_device() {
#if HAVE_CUDA
  if (restore_device_ != CPU_DEVICE_ID) {
    THCudaCheck(cudaSetDevice(restore_device_));
  }
#endif
}

} // namespace torch
} // namespace bluefog