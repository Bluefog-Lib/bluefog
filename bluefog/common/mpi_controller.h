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

#ifndef BLUEFOG_COMMON_MPI_CONTROLLER_H
#define BLUEFOG_COMMON_MPI_CONTROLLER_H

#include "logging.h"
#include "mpi_context.h"
#include "tensor_queue.h"
#include "timeline.h"

namespace bluefog {
namespace common {

// Function to check if the sending and receiving neighbors match in the
// topology.
bool CheckNeighborSendRecvPattern(int size, const TensorTableEntry& entry,
                                  Timeline* timeline_ptr, const MPI_Comm& comm);

class MPIController {
 public:
  MPIController(MPIContext& mpi_ctx) : mpi_ctx_(mpi_ctx) {
    BFLOG(DEBUG) << "MPI Controller Initialized.";
  }
  void Initialize();

  int GetTypeSize(DataType dtype);

  inline const std::vector<int>& GetRanks() const { return mpi_ctx_.ranks_; };
  inline int GetRank() const { return mpi_ctx_.rank_; };
  inline int GetLocalRank() const { return mpi_ctx_.local_rank_; };
  inline int GetCrossRank() const { return mpi_ctx_.cross_rank_; };
  inline int GetSize() const { return mpi_ctx_.size_; };
  inline int GetLocalSize() const { return mpi_ctx_.local_size_; };
  inline int GetCrossSize() const { return mpi_ctx_.cross_size_; };
  inline int GetNeighborSize() const { return mpi_ctx_.neighbor_indgree_; };
  inline const std::vector<int>& GetLocalCommRanks() const {
    return mpi_ctx_.local_comm_ranks_;
  };
  inline bool IsHomogeneous() const { return mpi_ctx_.is_homogeneous_; };

  inline bool IsMpiThreadsSupported() const { return mpi_threads_supported_; }
  inline bool IsWinObjectEmpty() const {
    return mpi_ctx_.named_win_map.size() == 0;
  }
  bool IsMpiUnifiedModel();

  // TODO(ybc) Create Operation_manager class to control it.
  void Allreduce(TensorTableEntry& entry);
  void Allgather(TensorTableEntry& entry);
  void Broadcast(TensorTableEntry& entry);
  void NeighborAllgather(TensorTableEntry& entry);
  void NeighborAllreduce(TensorTableEntry& entry);
  void PairGossip(TensorTableEntry& entry);

  void Allreduce(std::vector<TensorTableEntry>& entries);
  void NeighborAllreduce(std::vector<TensorTableEntry>& entries);

  void WinCreate(TensorTableEntry& entry);
  void WinFree(TensorTableEntry& entry);
  void WinFreeAll(TensorTableEntry& entry);

  void WinPut(TensorTableEntry& entry);
  void WinGet(TensorTableEntry& entry);
  void WinAccumulate(TensorTableEntry& entry);
  void Barrier(TensorTableEntry& entry);

  int SetTopology(int indegree, const int* sources, int outdegree,
                  const int* destinations);
  int SetTopologyWeights(int indegree, const int* sources,
                         double self_weight, const double* neighbor_weights);
  int LoadTopology(int* indegree, int*& sources, int* outdegree,
                   int*& destinations);
  int LoadTopologyWeights(double& self_weight,
                          const std::unordered_map<int, double>*& neighbor_weights);

  Status WinSync(const std::string& name, int device, bool with_associated_p);
  Status WinFence(const std::string& name);
  Status WinLock(const std::string& name);
  Status WinUnlock(const std::string& name);

  // This should be used by MPI Controller only.
  // Because NCCL controller uses MPI as mutex implementation as well.
  Status WinMutexAcquire(const std::string& name,
                         const std::vector<int>& acquire_ranks, bool is_sync);
  Status WinMutexRelease(const std::string& name,
                         const std::vector<int>& release_ranks, bool is_sync);
  Status WinVersionPutUpdate(const std::string& name, const std::vector<int>& ranks);
  Status WinVersionGetUpdate(const std::string& name, const std::vector<int>& ranks);
  Status VersionWinClear(const std::string& name);
  Status GetWindowVersionValue(const std::string& name, std::vector<int>& versions);

  Status GetWinAssociatedPByNameAndRank(const std::string& name, const int rank,
                                        double* weight);
  Status SetWinAssociatedPByNameAndRank(const std::string& name, const int rank,
                                        double weight);

 protected:
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

  // Outside dependencies
  MPIContext& mpi_ctx_;

  // flag indicating whether MPI multi-threading is supported.
  bool mpi_threads_supported_ = false;
};

// Our distributed mutex definition is different from the parallel computation
// concept. For a world size is N application, N mutex is created. Each
// process associates with one mutex. For window memory, we create independent
// local copies for each neighbor processes, so there is no conflict between
// the writing process from the neighbors (like win_put and win_accumulate).
// However, Win_sync (i.e update setup) will read it, which conflicted with
// other writting process. When WinMutexAcquire is called, we typically lock
// for all out-neighbors.
Status MPIWinMutexAcquireImpl(std::shared_ptr<MPI_Win> mutex_win,
                              const std::vector<int>& acquire_ranks,
                              int self_rank, bool is_sync);
Status MPIWinMutexReleaseImpl(std::shared_ptr<MPI_Win> mutex_win,
                              const std::vector<int>& release_ranks,
                              int self_rank, bool is_sync);

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_MPI_CONTROLLER_H
