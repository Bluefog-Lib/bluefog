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

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <nccl.h>
#include "cuda_runtime.h"
#include "mpi.h"
#include "thread_pool.h"

#include "common.h"
#include "logging.h"
#include "mpi_context.h"
#include "nccl_win.h"
#include "tensor_queue.h"
#include "timeline.h"

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

ncclDataType_t GetNCCLDataType(const DataType bf_data_type);
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

  void Initialize(const int rank, const int size,
                  const int local_rank);  // only initial nccl_comm etc.
  void Finalize();  // nccl_comm, peer, window will all be cleaned.
#if NCCL_MINOR < 7
  void CleanPeerCommunicators();
#endif
  void CleanWindowCommunicators();

  cudaError_t GetCudaEvent(cudaEvent_t* event);
  cudaError_t ReleaseCudaEvent(cudaEvent_t event);

  // TODO(ybc) Create intra-comm to allow the ops lik in-node allreduce.
  ncclComm_t nccl_comm;  // Store a global nccl comm.
  cudaStream_t stream;

  // We reuse CUDA events as it appears that their creation carries non-zero cost.
  std::queue<cudaEvent_t> cuda_events;
  std::mutex cuda_events_mutex;

  // Window Communicators. Because NCCL function is not thread-safe, each window
  // communication will be seperate by communicators.
  std::vector<ncclComm_t> nccl_win_comms;  // Same size as the world size.
  std::vector<cudaStream_t> nccl_win_streams;

#if NCCL_MINOR < 7
  // Communicators between two ranks used to mimic send/recv through broadcast.
  std::unordered_map<std::pair<int, int>, ncclComm_t, pair_hash>
      nccl_pair_comms = {};
  std::unordered_map<std::pair<int, int>, cudaStream_t, pair_hash>
      pair_streams = {};
  std::vector<std::pair<int, int>> pair_order = {};
#endif

  int cuda_device = -1;
  bool is_initialized = false;
  bool is_window_comm_initialized = false;
  bool is_peer_comm_initialized = false;

  ThreadPool finalizer_thread_pool;

  // Window related variables
  std::atomic_bool win_passive_recv_initialized{false};
  std::atomic_bool win_passive_recv_shutdown{false};
  mutable std::atomic_bool win_passive_recv_shutdown_done{false};
  std::thread win_passive_recv_thread;

  // Mimic MPI Windows used for one-sided communication. (Although there is no window). Manage
  // the persistent memory mainly.
  std::unordered_map<std::string, std::shared_ptr<NCCLWindowManager>> named_win_map;

  // In charge of mapping unique window name to id and reversed way.
  mutable NCCLWindowIdManager window_id_manager;
};

// Function to implement Window 
void WinPassiveRecvRequest(int self_rank, const NCCLContext& nccl_ctx);

class NCCLController {
 public:
  NCCLController(NCCLContext& nccl_ctx, MPIContext& mpi_ctx)
      : nccl_ctx_(nccl_ctx), mpi_ctx_(mpi_ctx) {
    BFLOG(DEBUG) << "NCCL Controller Initialized.";
  }

  // InitPeerCommunicator is always initialized when Initialize() is called.
  // But InitWindowCommunicator is independent since is only required when
  // window ops is used.
  void Initialize();
#if NCCL_MINOR < 7
  void InitPeerCommunicator();
  void DestroyPeerCommunicator();
#endif
  void InitWindowCommunicators();
  void DestroyWindowCommunicators();

  inline bool IsWinObjetEmpty() const {
    return nccl_ctx_.named_win_map.size() == 0;
  }

  void Allgather(TensorTableEntry& entry);
  void Allreduce(TensorTableEntry& entry);
  void Broadcast(TensorTableEntry& entry);
  void NeighborAllgather(TensorTableEntry& entry);
  void NeighborAllreduce(TensorTableEntry& entry);

  void WinPut(TensorTableEntry& entry);
  void WinGet(TensorTableEntry& entry);

  Status WinCreate(std::shared_ptr<Tensor> tensor,
                   std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                   const std::string& name, int device);

  Status WinFree(const std::string& name, int device);
  Status WinFreeAll();
  Status WinSync(const std::string& name, int device);

  Status WinMutexAcquire(const std::string& name,
                         const std::vector<int>& acquire_ranks, bool is_sync);
  Status WinMutexRelease(const std::string& name,
                         const std::vector<int>& release_ranks, bool is_sync);

 protected:
#if NCCL_MINOR < 7
  ncclResult_t ncclSendByBcast(const void* sendbuf, const int count,
                               ncclDataType_t data_type, int peer_rank);
  ncclResult_t ncclRecvByBcast(void* sendbuf, const int count,
                               ncclDataType_t data_type, int peer_rank);
#endif
 private:
  // Outside dependencies
  NCCLContext& nccl_ctx_;

  MPIContext& mpi_ctx_;

  Timeline* timeline_ptr_;
};

}  // namespace common
}  // namespace bluefog

#endif // BLUEFOG_COMMON_NCCL_CONTROLLER_H
