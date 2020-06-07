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

#include "nccl_controller.h"
#include "common.h"

namespace bluefog {
namespace common {

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
    case DataType::BLUEFOG_INT32:
      return ncclInt32;
    case DataType::BLUEFOG_INT64:
      return ncclInt64;
    case DataType::BLUEFOG_FLOAT16:
      return ncclFloat16;
    case DataType::BLUEFOG_FLOAT32:
      return ncclFloat32;
    case DataType::BLUEFOG_FLOAT64:
      return ncclFloat64;
    default:
      throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                             " is not supported in NCCL mode.");
  }
}

void NCCLContext::Initialize(const int rank, const int size,
                             const int local_rank) {
  if (is_initialized) {
    BFLOG(DEBUG)
        << "NCCL context has been initialized but NCCLContext::Initialize "
        << "is called again";
    return;
  }

  // A single rank will create a unique ID and send it to all other ranks to
  // make sure everyone has it.
  ncclUniqueId nccl_id;
  if (rank == 0) ncclGetUniqueId(&nccl_id);
  MPICHECK(
      MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Assume one device per process
  CUDACHECK(cudaSetDevice(local_rank));
  CUDACHECK(cudaStreamCreate(&s));

  ncclCommInitRank(&nccl_comm, size, nccl_id, rank);

  is_initialized = true;
  cuda_device = local_rank;
  return;
}

void NCCLContext::Finalize() {
  ncclCommDestroy(nccl_comm);
  is_initialized = false;
  cuda_device = -1;
}

void NCCLController::Initialize(const int rank, const int size,
                                const int local_rank) {
  nccl_ctx_.Initialize(rank, size, local_rank);
}

}  // namespace common
}  // namespace bluefog