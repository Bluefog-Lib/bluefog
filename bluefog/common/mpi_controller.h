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
    LOG(DEBUG) << "MPI Controller Initialized.";
  }
  void Initialize();
  
  int GetTypeSize(DataType dtype);

  std::vector<int>& GetRanks() { return ranks_; };
  int GetRank() { return rank_; };
  int GetLocalRank() { return local_rank_; };
  int GetCrossRank() { return cross_rank_; };
  int GetSize() { return size_; };
  int GetLocalSize() { return local_size_; };
  int GetCrossSize() { return cross_size_; };
  int GetNeighborSize() { return neighbor_indgree_; };
  const std::vector<int>& GetLocalCommRanks() { return local_comm_ranks_; };
  bool IsMpiThreadsSupported() const { return mpi_threads_supported_; }

  // TODO(ybc) Create Operation_manager class to control it.
  void Allreduce(TensorTableEntry& entries);
  void Allgather(TensorTableEntry& entries);
  void Broadcast(TensorTableEntry& entries);
  void NeighborAllgather(TensorTableEntry& entries);
  void NeighborAllreduce(TensorTableEntry& entries);
  void WinPut(TensorTableEntry& entries);
  void WinGet(TensorTableEntry& entries);

  int SetTopology(int indegree, const int* sources, int outdegree,
                  const int* destinations);
  int LoadTopology(int* indegree, int* sources, int* outdegree,
                   int* destinations);

  Status WinCreate(std::shared_ptr<Tensor> tensor,
                   std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                   const std::string& name, int device);
  Status WinFree(const std::string& name);
  Status WinSync(const std::string& name);

 protected:
  // Outside dependencies
  TensorQueue& tensor_queue_;

  MPIContext& mpi_ctx_;

  // flag indicating whether MPI multi-threading is supported
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
  int neighbor_is_weighted_ = -1;

  std::vector<int> neighbor_in_ranks_;
  std::vector<int> neighbor_out_ranks_;

  // ranks of the bluefog world
  std::vector<int> ranks_;

  // COMM_WORLD ranks of processes running on this node.
  std::vector<int> local_comm_ranks_;

};

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_MPI_CONTROLLER_H
