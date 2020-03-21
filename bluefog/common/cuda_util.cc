#if HAVE_CUDA
#include "cuda_runtime.h"
#endif

#include <stdio.h>
#include <assert.h>

#include "common.h"
#include "logging.h"
#include "cuda_util.h"

namespace bluefog {
namespace common {

#if HAVE_CUDA
#define CUDACHECK(ans) { CudaAssert((ans), __FILE__, __LINE__); }
inline void CudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

with_device::with_device(int device) {
  if (device == CPU_DEVICE_ID) {
    restore_device_ = CPU_DEVICE_ID;
  } else {
#if HAVE_CUDA
    CUDACHECK(cudaGetDevice(&restore_device_));
    CUDACHECK(cudaSetDevice(device));
#else
    throw std::logic_error("Internal error. Requested device context manager "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

with_device::~with_device() {
#if HAVE_CUDA
  if (restore_device_ != CPU_DEVICE_ID) {
    CUDACHECK(cudaSetDevice(restore_device_));
  }
#endif
}

} // namespace common
} // namespace bluefog