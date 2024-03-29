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

#if HAVE_CUDA
#include "cuda_runtime.h"
#endif

#include <assert.h>
#include <stdio.h>

#include "cuda_util.h"
#include "logging.h"

namespace bluefog {
namespace common {

with_device::with_device(int device) {
  if (device == CPU_DEVICE_ID) {
    restore_device_ = CPU_DEVICE_ID;
  } else {
#if HAVE_CUDA
    CUDACHECK(cudaGetDevice(&restore_device_));
    CUDACHECK(cudaSetDevice(device));
#else
    throw std::logic_error(
        "Internal error. Requested device context manager "
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

}  // namespace common
}  // namespace bluefog