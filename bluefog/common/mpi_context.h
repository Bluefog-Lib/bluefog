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

#ifndef BLUEFOG_COMMON_MPI_CONTEXT_H
#define BLUEFOG_COMMON_MPI_CONTEXT_H

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "mpi.h"

namespace bluefog {
namespace common {

// Base class for managing MPI environment.
class MPIContextManager {
 public:
  // Initialize MPI environment with required multi-threads support level.
  virtual void EnvInitialize(int mpi_threads_required);

  // Finalize MPI environment.
  virtual void EnvFinalize();
};

class WindowManager {
 public:
  WindowManager() = default;

  inline std::shared_ptr<MPI_Win> GetWinByRank(int rank) { 
    return wins_tensor_vec_[rank].first; 
  }
  inline std::shared_ptr<Tensor> GetAssociateTensorByRank(int rank) { 
    return wins_tensor_vec_[rank].second; 
  }
  inline std::shared_ptr<MPI_Win> GetGlobalWin() { return global_win_; }

  inline const void* GetWinMemoryByRank(int rank) { 
    return wins_tensor_vec_[rank].second->data(); 
  }

  inline void PushBackWinAndTensor(std::shared_ptr<MPI_Win> win,
                                   std::shared_ptr<Tensor> tensor) {
    wins_tensor_vec_.push_back(std::make_pair(win, tensor));
  }

  inline void SetGlobalWin(std::shared_ptr<MPI_Win> win) {
    global_win_ = win;
  }

  // Manually free the win memory.
  void FreeAllWins();

 private:
  // Store all the pointers to the MPI WIN and underlying tensor.
  // It should always keep the order from 0 to WORLD_SIZE-1.
  // Used with win_put.
  std::vector<std::pair<std::shared_ptr<MPI_Win>, std::shared_ptr<Tensor>>>
      wins_tensor_vec_;

  // A window associated with the self (all connected).
  // Used with win_accumulate and win_get.
  std::shared_ptr<MPI_Win> global_win_;
};

struct MPIContext {
  void Enable() { enabled_ = true; };
  void SetTopoSetup() { topo_setup_ = true; }
  void ResetTopoSetup() { topo_setup_ = false; }

  bool IsEnabled() { return enabled_; }
  bool IsTopoSetup() { return topo_setup_; }
  bool IsWeighted() { return is_weighted_; }
  void EnableTopoWeights() { is_weighted_ = true; }
  void DisableTopoWeights() { is_weighted_ = false; }

  // Take an argument of context manager pointer that will take care of
  // initialization of MPI environment.
  void Initialize(const std::vector<int>& ranks,
                  MPIContextManager& ctx_manager);

  // Take an argument of context manager pointer that will take care of
  // finalization of MPI environment.
  void Finalize(MPIContextManager& ctx_manager);
  MPI_Datatype GetMPIDataType(std::shared_ptr<Tensor> tensor);

  MPI_Datatype GetMPIDataType(DataType dtype);

  MPI_Comm GetMPICommunicator(Communicator comm);

  int GetMPITypeSize(DataType dtype);

  int BuildGraphComm(int indegree, const int* sources, int outdegree,
                     const int* destinations);

  bool RegisterWindowName(const std::string& name);
  std::shared_ptr<WindowManager> GetWindowByName(const std::string& name);
  bool UnregisterWindowName(const std::string& name);
  bool UnregisterAllWindowName();
  
  // The design of WinMutex is flawed. For example, if we call set topology within
  // the mutex acquire time, I don't know what will happen.
  bool InitializeWinMutex();
  bool DestroyWinMutex();

  // Flag indicating whether mpi is enabled.
  bool enabled_ = false;

  // Flag indicating whether mpi virtual topology is setup.
  bool topo_setup_ = false;
  bool is_weighted_ = false;

  // Private MPI communicator for Bluefog to ensure no collisions with other
  // threads using MPI.
  MPI_Comm mpi_comm;

  // Node-local communicator.
  MPI_Comm local_comm;

  // Cross-node communicator for hierarchical allreduce.
  MPI_Comm cross_comm;

  // Graph-based communicator for neighbor collective operations.
  MPI_Comm graph_comm;

  // MPI Windows used for one-sided communication.
  std::unordered_map<std::string, std::shared_ptr<WindowManager>> named_win_map;

  // MPI Window used for mutex.
  std::vector<std::shared_ptr<MPI_Win>> win_mutex;
  std::unique_ptr<int> self_mutex_mem = nullptr;

  // Whether mpi context should be finalize.
  bool should_finalize = false;
};

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_MPI_CONTEXT_H
