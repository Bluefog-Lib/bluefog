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

#include "mpi_context.h"
#include "tensor_queue.h"
#include "logging.h"

namespace bluefog {
namespace common {

class MPIController {
 public:
  MPIController(MPIContext& mpi_ctx)
      : mpi_ctx_(mpi_ctx) {
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

  inline bool IsMpiThreadsSupported() const { return mpi_threads_supported_; }
  inline bool IsWinObjetEmpty() const {
    return mpi_ctx_.named_win_map.size() == 0;
  }
  bool IsMpiUnifiedModel();

  // TODO(ybc) Create Operation_manager class to control it.
  void Allreduce(TensorTableEntry& entries);
  void Allgather(TensorTableEntry& entries);
  void Broadcast(TensorTableEntry& entries);
  void NeighborAllgather(TensorTableEntry& entries);
  void NeighborAllreduce(TensorTableEntry& entries);
  void WinPut(TensorTableEntry& entries);
  void WinGet(TensorTableEntry& entries);
  void WinAccumulate(TensorTableEntry& entries);
  void Barrier(TensorTableEntry& entry);

  int SetTopology(int indegree, const int* sources, int outdegree,
                  const int* destinations);
  int SetTopologyWeights(int indegree, const int* sources,
                         double self_weight, const double* neighbor_weights);
  int LoadTopology(int* indegree, int*& sources, int* outdegree,
                   int*& destinations);
  int LoadTopologyWeights(double& self_weight,
                          const std::unordered_map<int, double>*& neighbor_weights);

  Status WinCreate(std::shared_ptr<Tensor> tensor,
                   std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                   const std::string& name, int device);
  Status WinFree(const std::string& name, int device);
  Status WinFreeAll();
  Status WinSync(const std::string& name, int device);
  Status WinFence(const std::string& name);
  Status WinLock(const std::string& name);
  Status WinUnlock(const std::string& name);

  // Our distributed mutex definition is different from the parallel computation concept.
  // For a world size is N application, N mutex is created.
  // Each process associates with one mutex.
  // Note we create an independent local copy for neighbor process, so there is no conflict
  // between the writing process from the neighbors (like win_put and win_accumulate). However,
  // Win_sync (i.e update setup) will read it, which conflicted with other writting process.
  // When WinMutexAcquire is called, we typically lock for all out-neighbors.
  Status WinMutexAcquire(const std::vector<int>& acquire_ranks, bool is_sync);
  Status WinMutexRelease(const std::vector<int>& release_ranks, bool is_sync);

 protected:
  // Outside dependencies
  MPIContext& mpi_ctx_;

  // flag indicating whether MPI multi-threading is supported.
  bool mpi_threads_supported_ = false;
};

class WinMutexGuard {
 public:
  explicit WinMutexGuard(MPIController* mpi_controller,
                         const std::vector<int>& acquire_ranks,
                         bool is_sync);
  virtual ~WinMutexGuard();
  WinMutexGuard(const WinMutexGuard&) = delete;
  WinMutexGuard& operator=(const WinMutexGuard&) = delete;

 private:
  MPIController* const mpi_controller_;
  std::vector<int> acquire_ranks_;
  bool is_sync_;
};

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_MPI_CONTROLLER_H
