#ifndef BLUEFOG_COMMON_MPI_CONTEXT_H
#define BLUEFOG_COMMON_MPI_CONTEXT_H

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>

#include "mpi.h"
#include "common.h"

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

struct MPIContext {

  void Enable() {
    enabled_ = true;
  };

  bool IsEnabled() { return enabled_; }
  bool IsTopoSetup() { return topo_setup_; }

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

  bool AddNeighborWinMap(const std::string& name,
                             std::vector<std::shared_ptr<MPI_Win>> mpi_win_vec);

  // Flag indicating whether mpi is enabled.
  bool enabled_ = false;
  // Flag indicating whether mpi virtual topology is setup.
  // TODO(ybc) We need a topo finialized flag. After it turns ture, no more 
  // modification of topology.
  bool topo_setup_ = false;

  // Private MPI communicator for Bluefog to ensure no collisions with other
  // threads using MPI.
  MPI_Comm mpi_comm;

  // Node-local communicator.
  MPI_Comm local_comm;

  // Cross-node communicator for hierarchical allreduce.
  MPI_Comm cross_comm;

  // Graph-based communicator for neighbor collective operations.
  MPI_Comm graph_comm;

  // MPI Window used for one-sided communication
  std::unordered_map<std::string, std::shared_ptr<MPI_Win>> send_win_map;

  // MPI_Win for receiving the data from neighbors
  std::unordered_map<std::string, std::vector<std::shared_ptr<MPI_Win>>>
      neighbor_win_map;
  
  // MPI_Win with order. Because Win_free is collective ops.
  std::unordered_map<std::string, std::vector<std::shared_ptr<MPI_Win>>>
    win_map_with_order;

  // Whether mpi context should be finalize.
  bool should_finalize = false;
};

} // namespace common
} // namespace bluefog

#endif // BLUEFOG_COMMON_MPI_CONTEXT_H