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
  MPIController(TensorQueue& tensor_queue, MPIContext& mpi_ctx)
      : tensor_queue_(tensor_queue), mpi_ctx_(mpi_ctx) {
    BFLOG(DEBUG) << "MPI Controller Initialized.";
  }
  void Initialize();

  int GetTypeSize(DataType dtype);

  inline const std::vector<int>& GetRanks() const { return ranks_; };
  inline int GetRank() const { return rank_; };
  inline int GetLocalRank() const { return local_rank_; };
  inline int GetCrossRank() const { return cross_rank_; };
  inline int GetSize() const { return size_; };
  inline int GetLocalSize() const { return local_size_; };
  inline int GetCrossSize() const { return cross_size_; };
  inline int GetNeighborSize() const { return neighbor_indgree_; };
  inline const std::vector<int>& GetLocalCommRanks() const {
    return local_comm_ranks_;
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
                         float self_weight, const float* neighbor_weights);
  int LoadTopology(int* indegree, int*& sources, int* outdegree,
                   int*& destinations);
  int LoadTopologyWeights(float& self_weight,
                          const std::unordered_map<int, float>*& neighbor_weights);

  Status WinCreate(std::shared_ptr<Tensor> tensor,
                   std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                   const std::string& name, int device);
  Status WinFree(const std::string& name, int device);
  Status WinFreeAll();
  Status WinSync(const std::string& name, int device);
  Status WinFence(const std::string& name);
  Status WinLock(const std::string& name);
  Status WinUnlock(const std::string& name);

  // Our mutex definition is different from the parallel computation concept.
  // For a world size is N application, N mutex is created.
  // Each process associates with one mutex.
  // When WinMutexAcquire is called, we typically lock for all out-neighbors.
  // For example: if 1 want to win_put value to neighbor 0, and 2 want to
  // win_accumulate another value to neighbor 0 simultaneously, then only one rank 
  // can acquire the mutex to 0, i.e. two ops will be serialized.
  // It is most common used in win_sync (for self) and win_accumulate (for out-neighbor).
  Status WinMutexAcquire(const std::vector<int>& acquire_ranks);
  Status WinMutexRelease(const std::vector<int>& release_ranks);

 protected:
  // Outside dependencies
  TensorQueue& tensor_queue_;

  MPIContext& mpi_ctx_;

  // flag indicating whether MPI multi-threading is supported.
  bool mpi_threads_supported_ = false;

  Status AllocateOutput(TensorTableEntry& entries, int*& recvcounts, Communicator comm_type);
  void SetDisplacements(const int* recvcounts, int*& displcmnts, Communicator comm_type);

 private:
  int rank_ = 0;
  int local_rank_ = 0;

  int cross_rank_ = 0;
  int size_ = 1;
  int local_size_ = 1;
  int cross_size_ = 1;

  int neighbor_indgree_ = -1;
  int neighbor_outdgree_ = -1;

  std::vector<int> neighbor_in_ranks_;
  std::vector<int> neighbor_out_ranks_;

  // ranks of the bluefog world
  std::vector<int> ranks_;

  // COMM_WORLD ranks of processes running on this node.
  std::vector<int> local_comm_ranks_;

  float self_weight_;
  std::unordered_map<int, float> neighbor_weights_;
};

class WinMutexGuard {
 public:
  explicit WinMutexGuard(MPIController* mpi_controller,
                         const std::vector<int>& acquire_ranks);
  virtual ~WinMutexGuard();
  WinMutexGuard(const WinMutexGuard&) = delete;
  WinMutexGuard& operator=(const WinMutexGuard&) = delete;

 private:
  std::vector<int> acquire_ranks_;
  MPIController* const mpi_controller_;
};

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_MPI_CONTROLLER_H
