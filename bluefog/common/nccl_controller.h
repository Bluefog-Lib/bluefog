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

  void Initialize(const int rank, const int size, const int local_rank,
                  const int local_size, const MPI_Comm& world_comm,
                  const MPI_Comm& local_comm);  // only initial nccl_comm etc.
  void Finalize();  // nccl_comm, peer, window will all be cleaned.
#if NCCL_MINOR < 7
  void CleanPeerCommunicators();
#endif
  void CleanWindowCommunicators();

  cudaError_t GetCudaEvent(cudaEvent_t* event);
  cudaError_t ReleaseCudaEvent(cudaEvent_t event);

  ncclComm_t nccl_comm;  // Store a global nccl comm.
  ncclComm_t nccl_local_comm;  // Store a local nccl comm.
  cudaStream_t stream;

  // We reuse CUDA events as it appears that their creation carries non-zero cost.
  std::queue<cudaEvent_t> cuda_events;
  mutable std::mutex cuda_events_mutex;

  // Window Communicators. Because NCCL function is not thread-safe, each window
  // communication will be seperate by communicators.
  // New rank for self is always 0 in acitive comm and 1 in passive comm.
  std::vector<ncclComm_t> nccl_win_active_comms;  // Connect self is active and peer is passive.
  std::vector<ncclComm_t> nccl_win_passive_comms; // Connect self is passive and peer is active.

  std::vector<cudaStream_t> nccl_win_active_streams;
  std::vector<cudaStream_t> nccl_win_passive_streams;
  mutable std::mutex nccl_win_mutex;

#if NCCL_MINOR < 7
  // Communicators between two ranks used to mimic send/recv through broadcast.
  std::unordered_map<std::pair<int, int>, ncclComm_t, pair_hash>
      nccl_pair_comms = {};
  std::unordered_map<std::pair<int, int>, cudaStream_t, pair_hash>
      pair_streams = {};
  std::vector<std::pair<int, int>> pair_order = {};
#endif

  int cuda_device = -1;
  int self_rank = -1;
  int self_local_rank = -1;
  bool is_initialized = false;
  bool is_window_comm_initialized = false;
  bool is_peer_comm_initialized = false;

  mutable ThreadPool finalizer_thread_pool;

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
void WinPassiveRecvRequest(int self_rank, NCCLContext& nccl_ctx);

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
  void InitPeerCommunicators();
  void DestroyPeerCommunicators();
#endif
  void InitWindowCommunicators();
  void DestroyWindowCommunicators();

  inline bool IsWinObjectEmpty() const {
    return nccl_ctx_.named_win_map.size() == 0;
  }

  void Allgather(TensorTableEntry& entry);
  void Allreduce(TensorTableEntry& entry);
  void Broadcast(TensorTableEntry& entry);
  void NeighborAllgather(TensorTableEntry& entry);
  void NeighborAllreduce(TensorTableEntry& entry);

  void Allreduce(std::vector<TensorTableEntry>& entries);
  void NeighborAllreduce(std::vector<TensorTableEntry>& entries);

  void WinPut(TensorTableEntry& entry);
  void WinGet(TensorTableEntry& entry);
  void WinAccumulate(TensorTableEntry& entry);

  void WinCreate(TensorTableEntry& entry);
  void WinFree(TensorTableEntry& entry);
  void WinFreeAll(TensorTableEntry& entry);

  Status WinSync(const std::string& name, int device, bool with_associated_p);

  Status WinMutexAcquire(const std::string& name,
                         const std::vector<int>& acquire_ranks, bool is_sync);
  Status WinMutexRelease(const std::string& name,
                         const std::vector<int>& release_ranks, bool is_sync);

 protected:
  Status WinFreeReturnStatus(TensorTableEntry& entry);

#if NCCL_MINOR < 7
  ncclResult_t ncclSendByBcast(const void* sendbuf, const int count,
                               ncclDataType_t data_type, int peer_rank);
  ncclResult_t ncclRecvByBcast(void* sendbuf, const int count,
                               ncclDataType_t data_type, int peer_rank);
#endif

  void MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                            void*& buffer_data, size_t& buffer_len);

  void MemcpyOutFusionBuffer(const void* buffer_data,
                             std::vector<TensorTableEntry>& entries);

  void MemcpyOutFusionBufferForNeighbors(const void* buffer_data,
                                         std::vector<TensorTableEntry>& entries,
                                         const int num_recv_neighbors,
                                         const int64_t fused_data_size);

  void MemcpyOutFusionBufferForInputs(const void* fused_input_data,
                                      std::vector<TensorTableEntry>& entries);

  void MemcpyEntryInFusionBuffer(const TensorTableEntry& e,
                                 void* buffer_data_at_offset);

  void MemcpyEntryOutFusionBuffer(const void* buffer_data_at_offset,
                                  TensorTableEntry& e);

  void MemcpyEntryOutFusionBufferForNeighbors(const void* buffer_data_at_offset,
                                              TensorTableEntry& e,
                                              const int num_recv_neighbors,
                                              const int64_t fused_data_size);

  void RecordEvent(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                   std::string name);

  void WaitForEvents(
      std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
      const std::vector<TensorTableEntry>& entries, Timeline* timeline,
      const std::thread::id tid);

 private:
  // Outside dependencies
  NCCLContext& nccl_ctx_;

  MPIContext& mpi_ctx_;

  Timeline* timeline_ptr_;
};

}  // namespace common
}  // namespace bluefog

#endif // BLUEFOG_COMMON_NCCL_CONTROLLER_H
