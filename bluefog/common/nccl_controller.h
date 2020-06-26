// Copyright (C) 2020 Bluefog Team. All Rights Reserved.
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

#include <memory>
#include <utility>
#include <nccl.h>
#include "cuda_runtime.h"
#include "mpi.h"

#include "common.h"
#include "logging.h"
#include "mpi_context.h"
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

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Since we only used for a pair of int without many pairs,
        // this might be sufficient.
        return h1 ^ h2;  
    }
};

class NCCLContext {
 public:
  NCCLContext() = default;
  NCCLContext(const NCCLContext&) = delete;
  NCCLContext& operator=(NCCLContext other) = delete;

  void Initialize(const int rank, const int size, const int local_rank);
  void Finalize();
  void CleanPeerCommunicator();

  // TODO(ybc) Create e intra-comm to allow the ops lik in-node allreduce.
  ncclComm_t nccl_comm;  // Store a global nccl comm.
  cudaStream_t stream;

  // Communicators between two ranks used to mimic send/recv through broadcast.
  std::unordered_map<std::pair<int, int>, ncclComm_t, pair_hash>
      nccl_pair_comms = {};
  std::unordered_map<std::pair<int, int>, cudaStream_t, pair_hash>
      pair_streams = {};
  std::vector<std::pair<int, int>> pair_order = {};

  int cuda_device = -1;
  bool is_initialized = false;
  bool is_peer_initialized = false;
};

class NCCLController {
 public:
  NCCLController(NCCLContext& nccl_ctx, MPIContext& mpi_ctx)
      : nccl_ctx_(nccl_ctx), mpi_ctx_(mpi_ctx) {
    BFLOG(DEBUG) << "NCCL Controller Initialized.";
  }

  void Initialize();
#if NCCL_MINOR < 7
  void InitPeerCommunicator();
  void DestroyPeerCommunicator();
#endif

  void Allgather(TensorTableEntry& entries);
  void Allreduce(TensorTableEntry& entries);
  void Broadcast(TensorTableEntry& entries);
  void NeighborAllgather(TensorTableEntry& entries);
  void NeighborAllreduce(TensorTableEntry& entries);

 protected:
  ncclResult_t ncclSendByBcast(const void* sendbuf, const int count,
                               ncclDataType_t data_type, int peer_rank);
  ncclResult_t ncclRecvByBcast(void* sendbuf, const int count,
                               ncclDataType_t data_type, int peer_rank);

  // Outside dependencies
  NCCLContext& nccl_ctx_;

  MPIContext& mpi_ctx_;
};

}  // namespace common
}  // namespace bluefog

#endif // BLUEFOG_COMMON_NCCL_CONTROLLER_H
