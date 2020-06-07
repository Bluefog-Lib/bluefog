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
  CUDACHECK(cudaStreamCreate(&stream));

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

void NCCLController::Allreduce(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  void* buffer_data = (void*)entry.output->data();
  int num_elements = entry.tensor->shape().num_elements();

  int ret_code = ncclAllReduce(sendbuf, buffer_data, num_elements,
                               GetNCCLDataType(entry.tensor), ncclSum,
                               nccl_ctx_.nccl_comm, nccl_ctx_.stream);
  if (ret_code != ncclSuccess) {
    throw std::runtime_error(
        "ncclAllReduce failed, see NCCL output for details.");
  }
  // completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(nccl_ctx_.stream));
  entry.callback(Status::OK());
}

void NCCLController::Broadcast(TensorTableEntry& entry) {
  const int root_rank = entry.root_rank;
  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  const void* sendbuff;
  void* recvbuff;
  if (rank_ == root_rank) {
    sendbuff = entry.tensor->data();
    recvbuff = nullptr;
  } else {
    sendbuff = nullptr;
    recvbuff = (void*)entry.output->data();
  }
  int num_elements = entry.tensor->shape().num_elements();

  int ret_code = ncclBroadcast(sendbuf, buffer_data, num_elements, root_rank,
                               GetNCCLDataType(entry.tensor),
                               nccl_ctx_.nccl_comm, nccl_ctx_.stream);
  if (ret_code != ncclSuccess) {
    throw std::runtime_error("ncclBroadcast failed, see MPI output for details.");
  }
  // completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(nccl_ctx_.stream));
  entry.callback(Status::OK());
}

}  // namespace common
}  // namespace bluefog