#include <algorithm>
#include <cassert>
#include "mpi_controller.h"

namespace bluefog {
namespace common {

// MPIController
void MPIController::Initialize() {
  // Check if multi-thread is supported.
  int provided;
  MPI_Query_thread(&provided);
  mpi_threads_supported_ = (provided == MPI_THREAD_MULTIPLE);

  // Get MPI rank to determine if we are rank zero.
  MPI_Comm_rank(mpi_ctx_.mpi_comm, &rank_);

  // Get MPI size to determine how many tensors to wait for before reducing.
  MPI_Comm_size(mpi_ctx_.mpi_comm, &size_);

  // Determine local rank by querying the local communicator.
  MPI_Comm_rank(mpi_ctx_.local_comm, &local_rank_);
  MPI_Comm_size(mpi_ctx_.local_comm, &local_size_);
  local_comm_ranks_ = std::vector<int>((size_t)local_size_);
  local_comm_ranks_[local_rank_] = rank_;

  // Get cross-node rank and size in case of hierarchical allreduce.
  MPI_Comm_rank(mpi_ctx_.cross_comm, &cross_rank_);
  MPI_Comm_size(mpi_ctx_.cross_comm, &cross_size_);

  LOG(DEBUG) << "MPI controller initialized.";
}

int MPIController::GetTypeSize(DataType dtype) {
  return mpi_ctx_.GetMPITypeSize(dtype);
}

Status MPIController::AllocateOutput(TensorTableEntry& entry, int*& recvcounts,
                                     Communicator comm_type) {
  // Every tensor participating in Allgather operation may have different
  // first dimension size, but the rest of dimensions are same for all
  // tensors.  Here we get shape of tensor sliced by first dimension.
  // TODO(ybc): Check single_slice_shape is same cross all ranks.
  TensorShape single_slice_shape;
  for (int i = 1; i < entry.tensor->shape().dims(); ++i) {
    single_slice_shape.AddDim(entry.tensor->shape().dim_size(i));
  }

  // cnt_size is the number of expected receiving info.
  // For allgather, it is the global size.
  // For neighbor_allgather, it is the number of in-neighbor (excluding itself).
  int cnt_size = 0;
  if (comm_type == Communicator::GLOBAL) {
    cnt_size = size_;
  } else if (comm_type == Communicator::GRAPH) {
    cnt_size = neighbor_indgree_;
  }

  int* send_count = new int[1];
  send_count[0] = entry.tensor->shape().dim_size(0);
  int* gather_count = new int[cnt_size];
  int ret_code = -1;
  if (comm_type == Communicator::GLOBAL) {
    ret_code = MPI_Allgather(send_count, 1, MPI_INT, gather_count, 1, MPI_INT,
                             mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL));
  } else if (comm_type == Communicator::GRAPH) {
    ret_code = MPI_Neighbor_allgather(
        send_count, 1, MPI_INT, gather_count, 1, MPI_INT,
        mpi_ctx_.GetMPICommunicator(Communicator::GRAPH));
  }

  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Allgather (pre-allgather to get size) failed, see MPI output for "
        "details.");
  }

  // Copy tensor sizes from the response into a vector of int64_t
  // and compute total size.  This is size of first dimension.
  int64_t total_entry_dimension_size = 0;
  for (int rc = 0; rc < cnt_size; ++rc) {
    total_entry_dimension_size += gather_count[rc];
    recvcounts[rc] = single_slice_shape.num_elements() * gather_count[rc];
  }
  LOG(TRACE, rank_) << "total_entry_dimension_size: " << total_entry_dimension_size;

  // Allgather output will have shape of:
  // (sum of first dimension of every tensor) x (tensor slice shape).
  TensorShape output_shape;

  output_shape.AddDim((int64_t)total_entry_dimension_size);
  output_shape.AppendShape(single_slice_shape);

  Status status = entry.context->AllocateOutput(output_shape, &entry.output);
  return status;
}

void MPIController::SetDisplacements(const int* recvcounts, int*& displcmnts, Communicator comm_type) {
  int cnt_size = 0;
  if (comm_type == Communicator::GLOBAL) {
    cnt_size = size_;
  } else if (comm_type == Communicator::GRAPH) {
    cnt_size = neighbor_indgree_;
  }
  for (int rc = 0; rc < cnt_size; ++rc) {
    if (rc == 0) {
      displcmnts[rc] = 0;
    } else {
      displcmnts[rc] = displcmnts[rc - 1] + recvcounts[rc - 1];
    }
  }
}

void MPIController::Allgather(TensorTableEntry& entry) {
  int* recvcounts = new int[size_];
  int* displcmnts = new int[size_];
  AllocateOutput(entry, recvcounts, Communicator::GLOBAL);
  SetDisplacements(recvcounts, displcmnts, Communicator::GLOBAL);

  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();
  void* buffer_data = (void*)entry.output->data();

  int ret_code = MPI_Allgatherv(
      sendbuf, num_elements, mpi_ctx_.GetMPIDataType(entry.tensor), buffer_data,
      recvcounts, displcmnts, mpi_ctx_.GetMPIDataType(entry.output),
      mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL));
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Allgather failed, see MPI output for details.");
  }
  delete[] recvcounts;
  delete[] displcmnts;

  entry.callback(Status::OK());
}

void MPIController::Allreduce(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  void* buffer_data = (void*)entry.output->data();
  int num_elements = entry.tensor->shape().num_elements();
  int ret_code = MPI_Allreduce(sendbuf, buffer_data, num_elements,
                               mpi_ctx_.GetMPIDataType(entry.tensor), MPI_SUM,
                               mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL));
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_AllReduce failed, see MPI output for details.");
  }
  entry.callback(Status::OK());
}

void MPIController::Broadcast(TensorTableEntry& entry) {
  const int root_rank = entry.root_rank;
  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  void* data_ptr;
  if (rank_ == root_rank) {
    data_ptr = (void*) entry.tensor->data();
  } else {
    data_ptr = (void*) entry.output->data();
  } 
  int num_elements = entry.tensor->shape().num_elements();
  int ret_code =
      MPI_Bcast(data_ptr, num_elements, mpi_ctx_.GetMPIDataType(entry.tensor),
                root_rank, mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL));
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Bcast failed, see MPI output for details.");
  }
  entry.callback(Status::OK());
}

int MPIController::SetTopology(int indegree, const int* sources, int outdegree,
                               const int* destinations) {
  mpi_ctx_.BuildGraphComm(indegree, sources, outdegree, destinations);

  // Get neighbor in/out size and ranks.
  MPI_Dist_graph_neighbors_count(mpi_ctx_.graph_comm, &neighbor_indgree_,
                                 &neighbor_outdgree_, &neighbor_is_weighted_);

  neighbor_in_ranks_.clear();
  neighbor_in_ranks_.reserve(indegree);
  for (int i = 0; i < indegree; i++) {
    neighbor_in_ranks_.push_back(sources[i]);
  }

  neighbor_out_ranks_.clear();
  neighbor_out_ranks_.reserve(outdegree);
  for (int i = 0; i < outdegree; i++) {
    neighbor_out_ranks_.push_back(destinations[i]);
  }

  return 1;
}

int MPIController::LoadTopology(int* indegree, int* sources, int* outdegree,
                                int* destinations) {
  *indegree = neighbor_in_ranks_.size();
  *outdegree = neighbor_out_ranks_.size();
  sources = &neighbor_in_ranks_[0];
  destinations = &neighbor_out_ranks_[0];
  return 1;
}

void MPIController::NeighborAllgather(TensorTableEntry& entry) {
  int* recvcounts = new int[neighbor_indgree_];
  int* displcmnts = new int[neighbor_indgree_];
  if (!mpi_ctx_.IsTopoSetup()) {
    throw std::runtime_error("Topology of MPI has not been set yet.");
  }
  AllocateOutput(entry, recvcounts, Communicator::GRAPH);
  SetDisplacements(recvcounts, displcmnts, Communicator::GRAPH);

  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();
  void* buffer_data = (void*)entry.output->data();

  // Pitfall: mpi_neighbor_allgather do not include itself.
  int ret_code = MPI_Neighbor_allgatherv(
      sendbuf, num_elements, mpi_ctx_.GetMPIDataType(entry.tensor), buffer_data,
      recvcounts, displcmnts, mpi_ctx_.GetMPIDataType(entry.output),
      mpi_ctx_.GetMPICommunicator(Communicator::GRAPH));
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Neighbor_allgather failed, see MPI output for details.");
  }
  delete[] recvcounts;
  delete[] displcmnts;

  entry.callback(Status::OK());
}

void MPIController::NeighborAllreduce(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();

  // MPI have no neighbor_allreduce API. So we will utilize neighbor_allgather.
  // Allgather output will have shape of:
  // (sum of first dimension of every tensor) x (tensor slice shape).
  // For allreduce, the first dimension of every tensor should be the same.
  int total_entry_dimension_size =
      entry.tensor->shape().dim_size(0) * GetNeighborSize();
  TensorShape output_shape;
  output_shape.AddDim(total_entry_dimension_size);
  for (int i = 1; i < entry.tensor->shape().dims(); ++i) {
    output_shape.AddDim(entry.tensor->shape().dim_size(i));
  }
  Status status = entry.context->AllocateOutput(output_shape, &entry.output);
  void* buffer_data = (void*)entry.output->data();

  // Pitfall: Our neighbor_allreduce include itself, while
  // mpi_neighbor_allgather do not! Because for saving the communication there
  // is no need to transfer the local info again. However, for computation view,
  // including itself is more intuitive.
  int ret_code = MPI_Neighbor_allgather(
      sendbuf, num_elements, mpi_ctx_.GetMPIDataType(entry.tensor), buffer_data,
      num_elements, mpi_ctx_.GetMPIDataType(entry.output),
      mpi_ctx_.GetMPICommunicator(Communicator::GRAPH));
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Neighbor_allreduce(through neighbor_allgather) failed, see MPI "
        "output for details.");
  }

  entry.callback(Status::OK());
}

Status MPIController::WinCreate(std::shared_ptr<Tensor> tensor,
                                std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                                const std::string& name, const int device) {
  std::vector<std::shared_ptr<MPI_Win>> mpi_win_vec;
  int element_size = mpi_ctx_.GetMPITypeSize(tensor->dtype());
  int win_size = (tensor->shape().num_elements()) * element_size;
  int neighbor_tensor_index = 0;

  for (int rank = 0; rank < size_; rank++) {
    if (rank == rank_) {
      // Sender
      void* data_buf = (void*)tensor->data();

      auto mpi_win_ptr = std::make_shared<MPI_Win>();
      MPI_Win_create(data_buf, win_size, element_size, MPI_INFO_NULL,
                     mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL),
                     mpi_win_ptr.get());
      mpi_ctx_.send_win_map[name] = mpi_win_ptr;
      mpi_ctx_.win_map_with_order[name].push_back(mpi_win_ptr);
    } else if (std::find(neighbor_in_ranks_.begin(), neighbor_in_ranks_.end(),
                         rank) != neighbor_in_ranks_.end()) {
      // Receiver
      void* data_buf = (void*)neighbor_tensors[neighbor_tensor_index++]->data();
      auto mpi_win_ptr = std::make_shared<MPI_Win>();
      MPI_Win_create(data_buf, win_size, element_size, MPI_INFO_NULL,
                     mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL),
                     mpi_win_ptr.get());
      mpi_win_vec.push_back(mpi_win_ptr);
      mpi_ctx_.win_map_with_order[name].push_back(mpi_win_ptr);
    } else {
      // Just participate in a collective call.
      auto mpi_win_ptr = std::make_shared<MPI_Win>();
      MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL,
                     mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL),
                     mpi_win_ptr.get());
      mpi_ctx_.win_map_with_order[name].push_back(mpi_win_ptr);
    }
  }

  if (!mpi_ctx_.AddNeighborWinMap(name, mpi_win_vec)) {
    return Status::InvalidArgument(std::string("Win_create failed with ") +
                                   name);
  }
  return Status::OK();
}

Status MPIController::WinFree(const std::string& name) {
  auto it = mpi_ctx_.win_map_with_order.find(name);
  auto send_it = mpi_ctx_.send_win_map.find(name);
  auto neighbor_it = mpi_ctx_.neighbor_win_map.find(name);
  if (it == mpi_ctx_.win_map_with_order.end() ||
      send_it == mpi_ctx_.send_win_map.end() ||
      neighbor_it == mpi_ctx_.neighbor_win_map.end()) {
    return Status::InvalidArgument(std::string("Win_free failed with ") + name);
  }

  mpi_ctx_.send_win_map.erase(send_it);
  mpi_ctx_.neighbor_win_map.erase(neighbor_it);
  for (auto win_ptr : it->second) {
    MPI_Win_free(win_ptr.get());
  }
  mpi_ctx_.win_map_with_order.erase(it);

  return Status::OK();
}

Status MPIController::WinSync(const std::string& name) {
  auto neighbor_it = mpi_ctx_.neighbor_win_map.find(name);
  if (neighbor_it == mpi_ctx_.neighbor_win_map.end()) {
    return Status::InvalidArgument(std::string("Win_free failed with ") + name);
  }
  for (auto neighbor_win : neighbor_it->second) {
    MPI_Win_sync(*neighbor_win);
  }

  return Status::OK();
}

void MPIController::WinPut(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();
  MPI_Datatype data_type = mpi_ctx_.GetMPIDataType(entry.tensor);
  auto it = mpi_ctx_.send_win_map.find(entry.tensor_name);
  if (it == mpi_ctx_.send_win_map.end()) {
    throw std::runtime_error(std::string("Cannot find ") + entry.tensor_name);
  }
  MPI_Win mpi_win = *(it->second);
  
  int target_disp = 0; // offset in win buffer
  for (int target_rank : neighbor_out_ranks_) {
    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, MPI_MODE_NOCHECK, mpi_win);
    int ret_code = MPI_Put(sendbuf, num_elements, data_type, target_rank,
                           target_disp, num_elements, data_type, mpi_win);
    if (ret_code != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Put failed, see MPI output for details.");
    }
    MPI_Win_unlock(target_rank, mpi_win);
  }
  entry.callback(Status::OK());
}

}  // namespace common
}  // namespace bluefog
