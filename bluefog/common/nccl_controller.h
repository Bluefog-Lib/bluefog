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

#ifndef BLUEFOG_COMMON_NCCL_CONTROLLER_H
#define BLUEFOG_COMMON_NCCL_CONTROLLER_H

#include <nccl.h>
#include "mpi.h"

#include "common.h"
#include "logging.h"
#include "tensor_queue.h"

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

namespace bluefog {
namespace common {

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor);

struct NCCLContext {
  void Initialize(const int rank, const int size, const int local_rank);
  void Finalize();

  // TODO(ybc) Create e intra-comm to allow the ops lik in-node allreduce.
  ncclComm_t nccl_comm;  // Store a global nccl comm.
  cudaStream_t stream;

  int cuda_device = -1;
  bool is_initialized = false;
};

class NCCLController {
 public:
  NCCLController(TensorQueue& tensor_queue, NCCLContext& nccl_ctx)
      : tensor_queue_(tensor_queue), nccl_ctx_(nccl_ctx) {
    BFLOG(DEBUG) << "NCCL Controller Initialized.";
  }

  void Initialize(const int rank, const int size, const int local_rank);

  void Allreduce(TensorTableEntry& entries);
  void Broadcast(TensorTableEntry& entries);

 protected:
  // Outside dependencies
  TensorQueue& tensor_queue_;

  NCCLContext& nccl_ctx_;
};

}  // namespace common
}  // namespace bluefog

#endif BLUEFOG_COMMON_NCCL_CONTROLLER_H