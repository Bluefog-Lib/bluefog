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

  inline std::shared_ptr<MPI_Win> GetWinByRank(int rank) { return wins_[rank]; }

  inline void* GetWinMemoryByRank(int rank) { return win_memories_[rank]; }

  inline void PushBackWinAndMemory(std::shared_ptr<MPI_Win> win, void* memory) {
    wins_.push_back(win);
    win_memories_.push_back(memory);
  }

  // Manually free the win memory.
  void FreeAllWins();

 private:
  // Store all the pointers to the MPI WIN .
  // It should always keep the order from 0 to WORLD_SIZE-1.
  std::vector<std::shared_ptr<MPI_Win>> wins_;

  // Store all the underlying memories attached to the MPI WIN.
  // It should always keep the order from 0 to WORLD_SIZE-1.
  std::vector<void*> win_memories_;
};

struct MPIContext {
  void Enable() { enabled_ = true; };

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

  bool RegisterWindowName(const std::string& name, WindowManager& win_manager);
  bool UnregisterWindowName(const std::string& name);
  bool UnregisterAllWindowName();

  // Flag indicating whether mpi is enabled.
  bool enabled_ = false;
  // Flag indicating whether mpi virtual topology is setup.
  // TODO(ybc) We need a topo finialized flag. After it turns ture, no more
  // modification of topology.
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
  std::unordered_map<std::string, WindowManager> named_win_map;

  // Whether mpi context should be finalize.
  bool should_finalize = false;
};

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_MPI_CONTEXT_H