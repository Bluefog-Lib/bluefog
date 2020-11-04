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

#include "mpi_controller.h"

#if HAVE_CUDA
#include "cuda_runtime.h"
#endif

#include <algorithm>
#include <cassert>
#include <cstring>
#include <thread>

#include "cuda_util.h"
#include "operations.h"
#include "timeline.h"

namespace bluefog {
namespace common {

// It may be because the win_create is called at different
// threads from the win_put, win_get, etc. After moving win_create into
// communicaiton thread, it resolved. (works in Openmpi=4.0.2 and MPICH).
// Due to unclear reason that mpi_put/get/accumlate under the
// mpi_lock epoch cannot send too long vector in one time, we
// define this number as the maximum size of win_ops can send.
static const char* BLUEFOG_MAX_WIN_SENT =
    std::getenv("BLUEFOG_MAX_WIN_SENT_LENGTH");
static const int MAX_WIN_SENT =
    BLUEFOG_MAX_WIN_SENT == nullptr
        ? 1000
        : std::strtol(BLUEFOG_MAX_WIN_SENT, nullptr, 10);

// MPIController
void MPIController::Initialize() {
  // Check if multi-thread is supported.
  int provided;
  MPI_Query_thread(&provided);
  mpi_threads_supported_ = (provided == MPI_THREAD_MULTIPLE);

  // Get MPI rank to determine if we are rank zero.
  MPI_Comm_rank(mpi_ctx_.mpi_comm, &mpi_ctx_.rank_);

  // Get MPI size to determine how many tensors to wait for before reducing.
  MPI_Comm_size(mpi_ctx_.mpi_comm, &mpi_ctx_.size_);

  // Determine local rank by querying the local communicator.
  MPI_Comm_rank(mpi_ctx_.local_comm, &mpi_ctx_.local_rank_);
  MPI_Comm_size(mpi_ctx_.local_comm, &mpi_ctx_.local_size_);
  mpi_ctx_.local_comm_ranks_ = std::vector<int>((size_t)mpi_ctx_.local_size_);
  mpi_ctx_.local_comm_ranks_[mpi_ctx_.local_rank_] = mpi_ctx_.rank_;

  // Get cross-node rank and size in case of hierarchical allreduce.
  MPI_Comm_rank(mpi_ctx_.cross_comm, &mpi_ctx_.cross_rank_);
  MPI_Comm_size(mpi_ctx_.cross_comm, &mpi_ctx_.cross_size_);

  // Determine if cluster is homogeneous, i.e., if every node has the same
  // local_size
  auto local_sizes = std::vector<int>(mpi_ctx_.size_);
  MPI_Allgather(&mpi_ctx_.local_size_, 1, MPI_INT, local_sizes.data(), 1, MPI_INT,
                mpi_ctx_.mpi_comm);

  mpi_ctx_.is_homogeneous_ = true;
  for (int i = 0; i < mpi_ctx_.size_; ++i) {
    if (local_sizes[i] != mpi_ctx_.local_size_) {
      mpi_ctx_.is_homogeneous_ = false;
      break;
    }
    // also require the rank is continuous
    if (mpi_ctx_.rank_ % local_sizes[i] != mpi_ctx_.local_rank_) {
      mpi_ctx_.is_homogeneous_ = false;
      break;
    }
  }
  BFLOG(TRACE) << "Running environment " << (mpi_ctx_.is_homogeneous_ ? "is" : "is NOT")
               << " homogeneous (i.e. same local size on each machine)";
  if (!mpi_ctx_.is_homogeneous_) {
    BFLOG(WARNING)
        << "Running environment is not homogeneous (i.e. same local size on "
           "each machine), which may disable some functionality or degrade "
           "performance.";
  }

  BFLOG(DEBUG) << "MPI controller initialized.";
}

int MPIController::GetTypeSize(DataType dtype) {
  return mpi_ctx_.GetMPITypeSize(dtype);
}

void MPIController::Allgather(TensorTableEntry& entry) {
  int* recvcounts = new int[mpi_ctx_.size_];
  int* displcmnts = new int[mpi_ctx_.size_];
  Status status = mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GLOBAL);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GLOBAL);
  if (!status.ok()) {
    delete[] recvcounts;
    delete[] displcmnts;
    entry.callback(status);
    return;
  }

  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();
  void* buffer_data = (void*)entry.output->data();

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

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
  const void* sendbuf = entry.tensor->data() == entry.output->data()
                            ? MPI_IN_PLACE
                            : entry.tensor->data();
  void* buffer_data = (void*)entry.output->data();
  int num_elements = entry.tensor->shape().num_elements();

  // Here is_hierarchical == true means local allreduce.
  auto communicator_type =
      entry.is_hierarchical ? Communicator::LOCAL : Communicator::GLOBAL;

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);
  int ret_code = MPI_Allreduce(sendbuf, buffer_data, num_elements,
                               mpi_ctx_.GetMPIDataType(entry.tensor),
                               mpi_ctx_.GetMPISumOp(entry.tensor->dtype()),
                               mpi_ctx_.GetMPICommunicator(communicator_type));
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
  if (mpi_ctx_.rank_ == root_rank) {
    data_ptr = (void*)entry.tensor->data();
  } else {
    data_ptr = (void*)entry.output->data();
  }
  int num_elements = entry.tensor->shape().num_elements();

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);
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
  mpi_ctx_.ResetTopoSetup();
  int res_build_graph =
      mpi_ctx_.BuildGraphComm(indegree, sources, outdegree, destinations);
  if (res_build_graph == -1) return -1;
  mpi_ctx_.SetTopoSetup();

  // Get neighbor in/out size and ranks.
  int unused_neighbor_is_weighted_ = -1;
  MPI_Dist_graph_neighbors_count(mpi_ctx_.graph_comm, &mpi_ctx_.neighbor_indgree_,
                                 &mpi_ctx_.neighbor_outdgree_,
                                 &unused_neighbor_is_weighted_);

  // Clear the previous neighbor_in_ranks_ is necessary because we might
  // change the topology.
  mpi_ctx_.neighbor_in_ranks_.clear();
  mpi_ctx_.neighbor_in_ranks_.reserve(indegree);
  for (int i = 0; i < indegree; i++) {
    mpi_ctx_.neighbor_in_ranks_.push_back(sources[i]);
  }

  mpi_ctx_.neighbor_out_ranks_.clear();
  mpi_ctx_.neighbor_out_ranks_.reserve(outdegree);
  for (int i = 0; i < outdegree; i++) {
    mpi_ctx_.neighbor_out_ranks_.push_back(destinations[i]);
  }
  mpi_ctx_.DisableTopoWeights();  // Topology weights are always set at
                                  // SetTopologyWeights.
  return 1;
}

int MPIController::SetTopologyWeights(int indegree, const int* sources,
                                      double self_weight, const double* neighbor_weights) {
  // We assume when this function is called, the base topology has already
  // been set. Here the neighbor_weights specifies the weights from the sources.
  if (!mpi_ctx_.IsTopoSetup()) {
    return -1;
  }
  mpi_ctx_.self_weight_ = self_weight;
  for (int i = 0; i < indegree; i++) {
    mpi_ctx_.neighbor_weights_[sources[i]] = neighbor_weights[i];
  }
  mpi_ctx_.EnableTopoWeights();
  return 1;
}

int MPIController::LoadTopology(int* indegree, int*& sources, int* outdegree,
                                int*& destinations) {
  *indegree = mpi_ctx_.neighbor_in_ranks_.size();
  sources = &mpi_ctx_.neighbor_in_ranks_[0];
  *outdegree = mpi_ctx_.neighbor_out_ranks_.size();
  destinations = &mpi_ctx_.neighbor_out_ranks_[0];
  return 1;
}

int MPIController::LoadTopologyWeights(
    double& self_weight,
    const std::unordered_map<int, double>*& neighbor_weights) {
  if (!mpi_ctx_.IsWeighted()) {
    return 0;
  }
  self_weight = mpi_ctx_.self_weight_;
  neighbor_weights = &mpi_ctx_.neighbor_weights_;
  return 1;
}

void MPIController::NeighborAllgather(TensorTableEntry& entry) {
  int* recvcounts = new int[mpi_ctx_.neighbor_indgree_];
  int* displcmnts = new int[mpi_ctx_.neighbor_indgree_];
  if (!mpi_ctx_.IsTopoSetup()) {
    throw std::runtime_error("Topology of MPI has not been set yet.");
  }
  Status status = mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GRAPH);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GRAPH);
  if (!status.ok()) {
    delete[] recvcounts;
    delete[] displcmnts;
    entry.callback(status);
    return;
  }

  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();
  void* buffer_data = (void*)entry.output->data();

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
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
  timeline_ptr->ActivityEnd(entry.tensor_name);

  timeline_ptr->ActivityStart(entry.tensor_name, "CALLBACK");
  entry.callback(Status::OK());
  timeline_ptr->ActivityEnd(entry.tensor_name);
}

// Function to check if the sending and receiving neighbors match in the topology.
bool CheckNeighborSendRecvPattern(int size, const TensorTableEntry& entry,
                                  Timeline* timeline_ptr, const MPI_Comm& comm) {
  bool res = false;
  // enabled the check if enable_topo_check is true and partial
  // neighbor_allreduce is activated.
  if (entry.enable_topo_check && entry.dynamic_neighbors_enabled) {
    if (entry.is_hierarchical) {
      // TODO: support check.
      BFLOG(INFO) << "Request to check topology for hierarchical neighbor "
                  << "allreduce ops but it is not supported yet.";
      return res;
    }
    timeline_ptr->ActivityStart(entry.tensor_name, "NEGOTIATION");
    // Put all the send and recv neighbors in a single vector, and obtain a send
    // matrix and a recv matrix through MPI_Allgather.
    bool* send_check_buf = new bool[2 * size];
    std::fill_n(send_check_buf, 2 * size, false);
    bool* recv_check_buf = new bool[2 * size * size];
    for (int send_rank : *(entry.send_neighbors))
      send_check_buf[send_rank] = true;
    for (int recv_rank : *(entry.recv_neighbors))
      send_check_buf[size + recv_rank] = true;
    int ret_code = MPI_Allgather(send_check_buf, size * 2, MPI_C_BOOL,
                                 recv_check_buf, size * 2, MPI_C_BOOL, comm);
    if (ret_code != MPI_SUCCESS) {
      throw std::runtime_error(
          "MPI_Allgather (for dynamic neighbor_allreduce negotiation) failed, "
          "see MPI output for details.");
    }
    // This checks that send matrix and transposed recv matrix should be the
    // same. If same, the topology is good to go. If not, there is mismatch edge
    // to be fixed.
    auto GetSendIndex = [size](int i, int j) -> int { return 2*size*i+j; };
    auto GetRecvIndex = [size](int i, int j) -> int { return 2*size*i+j+size; };
    for (int i = 0; i < size; ++i) {
      if (res) break;
      for (int j = 0; j < size; ++j) {
        if (recv_check_buf[GetSendIndex(i, j)] !=
            recv_check_buf[GetRecvIndex(j, i)]) {
          res = true;
          break;
        }
      }
    }
    delete [] send_check_buf;
    delete [] recv_check_buf;
    timeline_ptr->ActivityEnd(entry.tensor_name);
  }
  return res;
}

std::string GenerateNeighborAllreduceErrorMessage(const std::vector<MPI_Status>& statuses,
                                                  int nsend, int nrecv) {
  std::string error_message = "";
  bool error_encountered = false;
  for (int i = 0; i < nsend; ++i) {
    const auto& status = statuses[i];
    error_message += "MPI_Isend to Process " + std::to_string(status.MPI_SOURCE);
    error_message += "; with tag " + std::to_string(status.MPI_TAG);
    error_message += "; with error code " + std::to_string(status.MPI_ERROR) + "\n";
    if(status.MPI_ERROR != MPI_SUCCESS) error_encountered = true;
  }
  for (int i = 0; i < nrecv; ++i) {
    const auto& status = statuses[i+nsend];
    error_message += "MPI_Irecv from Process " + std::to_string(status.MPI_SOURCE);
    error_message += "; with tag " + std::to_string(status.MPI_TAG);
    error_message += "; with error code " + std::to_string(status.MPI_ERROR) + "\n";
    if(status.MPI_ERROR != MPI_SUCCESS) error_encountered = true;
  }
  if (!error_encountered) error_message = "";
  return error_message;
}

void MPIController::NeighborAllreduce(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  // MPI have no neighbor_allreduce API. So we will utilize neighbor_allgather.
  // Allgather output will have shape of:
  // (sum of first dimension of every tensor) x (tensor slice shape).
  // For allreduce, the first dimension of every tensor should be the same.
  // Assume the memory has already been allocated at python side.
  void* buffer_data = (void*)entry.output->data();

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  // If only partial sending is enabled, the following code block checks whether the sending
  // and recieving neighbors match each other when enable_topo_check is set to be True.
  bool is_topo_check_fail = CheckNeighborSendRecvPattern(
      mpi_ctx_.size_, entry, timeline_ptr,
      mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL));

  if (is_topo_check_fail) {
    entry.callback(Status::InvalidArgument(
        "Send and recv neighbors dont' match in neighbor "
        "allreduce with partial send/recv request."));
    return;
  }

  timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
  // Pitfall: Our neighbor_allreduce include itself, while
  // mpi_neighbor_allgather do not! Because for saving the communication there
  // is no need to transfer the local info again. However, for computation view,
  // including itself is more intuitive.
  std::string error_message = "";

  if (!entry.is_hierarchical) {
    if (!entry.dynamic_neighbors_enabled) {
      int ret_code = MPI_Neighbor_allgather(
          sendbuf, num_elements, mpi_ctx_.GetMPIDataType(entry.tensor),
          buffer_data, num_elements, mpi_ctx_.GetMPIDataType(entry.output),
          mpi_ctx_.GetMPICommunicator(Communicator::GRAPH));
      if (ret_code != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Neighbor_allreduce (through neighbor_allgather) failed, see "
            "MPI "
            "output for details.");
      }
    } else {
      int nsend = entry.send_neighbors->size();
      int nrecv = entry.recv_neighbors->size();
      std::vector<MPI_Request> requests(nsend + nrecv);
      std::vector<MPI_Status> statuses(nsend + nrecv);
      int element_size = mpi_ctx_.GetMPITypeSize(entry.output->dtype());
      for (int i = 0; i < nrecv; ++i) {
        void* recvbuf = (void*)(static_cast<const char*>(entry.output->data()) +
                                num_elements * i * element_size);
        int ret_code = MPI_Irecv(
            recvbuf, num_elements, mpi_ctx_.GetMPIDataType(entry.output),
            entry.recv_neighbors->at(i),
            mpi_ctx_.rank_ + entry.recv_neighbors->at(i),
            mpi_ctx_.GetMPICommunicator(Communicator::GRAPH),
            &requests[i + nsend]);
        if (ret_code != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI_Irecv (for dynamic neighbor_allreduce) failed, see MPI "
              "output for details.");
        }
      }
      for (int i = 0; i < nsend; ++i) {
        int ret_code = MPI_Isend(
            sendbuf, num_elements, mpi_ctx_.GetMPIDataType(entry.tensor),
            entry.send_neighbors->at(i),
            mpi_ctx_.rank_ + entry.send_neighbors->at(i),
            mpi_ctx_.GetMPICommunicator(Communicator::GRAPH), &requests[i]);
        if (ret_code != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI_Isend (for dynamic neighbor_allreduce) failed, see MPI "
              "output for details.");
        }
      }
      MPI_Waitall(nsend + nrecv, requests.data(), statuses.data());
      error_message =
          GenerateNeighborAllreduceErrorMessage(statuses, nsend, nrecv);
    }
  } else {
    if (entry.send_neighbors->empty()) {
      throw std::runtime_error(
          "Under hierarchical neighbor_allreduce, argument "
          "send_machine_neighbors should not be empty.");
    }
    if (mpi_ctx_.local_size_ < 2) {
      throw std::runtime_error(
          "Local size is smaller than 2, in this case, you should use "
          "neighbor_allreduce instead of hierarchical_neighbor_allreduce.");
    }
    // 1. In-place allreduce
    MPI_Allreduce(MPI_IN_PLACE, (void*)sendbuf, num_elements,
                  mpi_ctx_.GetMPIDataType(entry.tensor), MPI_SUM,
                  mpi_ctx_.GetMPICommunicator(Communicator::LOCAL));
    // 2. Local_rank = 0 do the neighbor all with other machines local_rank=0.
    if (mpi_ctx_.local_rank_ == 0) {
      int nsend = entry.send_neighbors->size();
      int nrecv = entry.recv_neighbors->size();
      std::vector<MPI_Request> requests(nsend + nrecv);
      std::vector<MPI_Status> statuses(nsend + nrecv);
      int element_size = mpi_ctx_.GetMPITypeSize(entry.output->dtype());
      for (int i = 0; i < nrecv; ++i) {
        void* recvbuf = (void*)(static_cast<const char*>(entry.output->data()) +
                                num_elements * i * element_size);
        int ret_code = MPI_Irecv(
            recvbuf, num_elements, mpi_ctx_.GetMPIDataType(entry.output),
            entry.recv_neighbors->at(i),
            mpi_ctx_.rank_ + entry.recv_neighbors->at(i),
            mpi_ctx_.GetMPICommunicator(Communicator::GRAPH),
            &requests[i + nsend]);
        if (ret_code != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI_Irecv (for dynamic neighbor_allreduce) failed, see MPI "
              "output for details.");
        }
      }
      for (int i = 0; i < nsend; ++i) {
        int ret_code = MPI_Isend(
            sendbuf, num_elements, mpi_ctx_.GetMPIDataType(entry.tensor),
            entry.send_neighbors->at(i),
            mpi_ctx_.rank_ + entry.send_neighbors->at(i),
            mpi_ctx_.GetMPICommunicator(Communicator::GRAPH), &requests[i]);
        if (ret_code != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI_Isend (for dynamic neighbor_allreduce) failed, see MPI "
              "output for details.");
        }
      }
      MPI_Waitall(nsend + nrecv, requests.data(), statuses.data());
      error_message =
          GenerateNeighborAllreduceErrorMessage(statuses, nsend, nrecv);
    } else {
      // Do nothing here.
    }
    // 3. Broadcast recv data from local rank = 0 to other local ranks.
    int recv_num_elements = num_elements * entry.recv_neighbors->size();
    MPI_Bcast(buffer_data, recv_num_elements,
              mpi_ctx_.GetMPIDataType(entry.output), 0,
              mpi_ctx_.GetMPICommunicator(Communicator::LOCAL));
  }
  timeline_ptr->ActivityEnd(entry.tensor_name);

  timeline_ptr->ActivityStart(entry.tensor_name, "COMPUTE_AVERAGE");
  if (error_message != "") {
    entry.callback(Status::UnknownError(error_message));
  } else { 
    entry.callback(Status::OK());
  }
  timeline_ptr->ActivityEnd(entry.tensor_name);
}

void MPIController::Allreduce(std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  with_device device_guard(first_entry.device);

  void* buffer_data;
  size_t buffer_len = 0;
  int64_t num_elements = 0;
  for (auto& e : entries) {
    num_elements += e.tensor->shape().num_elements();
  }
  Timeline* timeline_ptr;
  GetBluefogTimeline(timeline_ptr);

  timeline_ptr->ActivityStartAll(entries, "MEMCPY_IN_FUSION_BUFFER");
  MemcpyInFusionBuffer(entries, buffer_data, buffer_len);
  timeline_ptr->ActivityEndAll(entries);

  timeline_ptr->ActivityStartAll(entries, "COMMUNICATE");
  // Here is_hierarchical == true means local allreduce.
  auto communicator_type =
      first_entry.is_hierarchical ? Communicator::LOCAL : Communicator::GLOBAL;
  int ret_code =
      MPI_Allreduce(MPI_IN_PLACE, buffer_data, num_elements,
                    mpi_ctx_.GetMPIDataType(first_entry.tensor),
                    mpi_ctx_.GetMPISumOp(first_entry.tensor->dtype()),
                    mpi_ctx_.GetMPICommunicator(communicator_type));
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_AllReduce failed, see MPI output for details.");
  }
  timeline_ptr->ActivityEndAll(entries);

  timeline_ptr->ActivityStartAll(entries, "MEMCPY_OUT_FUSION_BUFFER");
  MemcpyOutFusionBuffer(buffer_data, entries);
  timeline_ptr->ActivityEndAll(entries);

  for (auto& e : entries) {
    e.callback(Status::OK());
  }
}

// TODO: reuse the code of NeighborAllreduce without fusion.
void MPIController::NeighborAllreduce(std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  with_device device_guard(first_entry.device);

  void* buffer_data;
  size_t buffer_len = 0;
  int64_t num_elements = 0;
  for (auto& e : entries) {
    num_elements += e.tensor->shape().num_elements();
  }
  int element_size = mpi_ctx_.GetMPITypeSize(first_entry.tensor->dtype());
  Timeline* timeline_ptr;
  GetBluefogTimeline(timeline_ptr);

  // If only partial sending is enabled, the following code block checks whether
  // the sending and recieving neighbors match each other when enable_topo_check
  // is set to be True.
  bool is_topo_check_fail = CheckNeighborSendRecvPattern(
      mpi_ctx_.size_, first_entry, timeline_ptr,
      mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL));

  if (is_topo_check_fail) {
    for (auto& entry : entries) {
      entry.callback(Status::InvalidArgument(
          "Send and recv neighbors dont' match in neighbor "
          "allreduce with partial send/recv request."));
    }
    return;
  }

  timeline_ptr->ActivityStartAll(entries, "MEMCPY_IN_FUSION_BUFFER");
  MemcpyInFusionBuffer(entries, buffer_data, buffer_len);
  timeline_ptr->ActivityEndAll(entries);
  const void* fused_input_data = buffer_data;

  // Unlike allreduce, the storage for neighbor_allreduce in fusion buffer
  // is like [t_1, t_2 | t_1_n1, t_2_n1, t_1_n2, t_2_n2].
  // Here t_1 and t_2  means self tensor 1 and 2 and _n1 and _n2 means the
  // recieving tensors for neighbor 1 and 2;
  // Hence, we need to offset the buffer data to location for neighbors.
  buffer_data = (uint8_t*)buffer_data + num_elements * element_size;

  timeline_ptr->ActivityStartAll(entries, "COMMUNICATE");
  // Pitfall: Our neighbor_allreduce include itself, while
  // mpi_neighbor_allgather do not! Because for saving the communication there
  // is no need to transfer the local info again. However, for computation view,
  // including itself is more intuitive.
  std::string error_message = "";

  if (!first_entry.is_hierarchical) {
    if (!first_entry.dynamic_neighbors_enabled) {
      int ret_code = MPI_Neighbor_allgather(
          fused_input_data, num_elements, mpi_ctx_.GetMPIDataType(first_entry.tensor),
          buffer_data, num_elements, mpi_ctx_.GetMPIDataType(first_entry.output),
          mpi_ctx_.GetMPICommunicator(Communicator::GRAPH));
      if (ret_code != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Neighbor_allreduce (through neighbor_allgather) failed, see MPI "
            "output for details.");
      }
    } else {
      int nsend = first_entry.send_neighbors->size();
      int nrecv = first_entry.recv_neighbors->size();
      std::vector<MPI_Request> requests(nsend + nrecv);
      std::vector<MPI_Status> statuses(nsend + nrecv);
      for (int i = 0; i < nrecv; ++i) {
        void* recvbuf =
            (void*)((uint8_t*)buffer_data + num_elements * i * element_size);
        int ret_code = MPI_Irecv(recvbuf, num_elements,
                                mpi_ctx_.GetMPIDataType(first_entry.output),
                                first_entry.recv_neighbors->at(i),
                                /*tag=*/mpi_ctx_.rank_ + first_entry.recv_neighbors->at(i),
                                mpi_ctx_.GetMPICommunicator(Communicator::GRAPH),
                                &requests[i + nsend]);
        if (ret_code != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI_Irecv (for dynamic neighbor_allreduce) failed, see MPI output "
              "for details.");
        }
      }
      for (int i = 0; i < nsend; ++i) {
        int ret_code = MPI_Isend(
            fused_input_data, num_elements, mpi_ctx_.GetMPIDataType(first_entry.tensor),
            first_entry.send_neighbors->at(i),
            /*tag=*/mpi_ctx_.rank_ + first_entry.send_neighbors->at(i),
            mpi_ctx_.GetMPICommunicator(Communicator::GRAPH), &requests[i]);
        if (ret_code != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI_Isend (for dynamic neighbor_allreduce) failed, see MPI output "
              "for details.");
        }
      }
      MPI_Waitall(nsend + nrecv, requests.data(), statuses.data());
      error_message =
          GenerateNeighborAllreduceErrorMessage(statuses, nsend, nrecv);
    }
  } else {
    if (first_entry.send_neighbors->empty()) {
      throw std::runtime_error(
          "Under hierarchical neighbor_allreduce, argument "
          "send_machine_neighbors should not be empty.");
    }
    if (mpi_ctx_.local_size_ < 2) {
      throw std::runtime_error(
          "Local size is smaller than 2, in this case, you should use "
          "neighbor_allreduce instead of hierarchical_neighbor_allreduce.");
    }
    // 1. In-place allreduce
    MPI_Allreduce(MPI_IN_PLACE, (void*)fused_input_data, num_elements,
                  mpi_ctx_.GetMPIDataType(first_entry.tensor), MPI_SUM,
                  mpi_ctx_.GetMPICommunicator(Communicator::LOCAL));
    // 2. Local_rank = 0 do the neighbor all with other machines local_rank=0.
    if (mpi_ctx_.local_rank_ == 0) {
      int nsend = first_entry.send_neighbors->size();
      int nrecv = first_entry.recv_neighbors->size();
      std::vector<MPI_Request> requests(nsend + nrecv);
      std::vector<MPI_Status> statuses(nsend + nrecv);
      for (int i = 0; i < nrecv; ++i) {
        void* recvbuf =
            (void*)((uint8_t*)buffer_data + num_elements * i * element_size);
        int ret_code = MPI_Irecv(recvbuf, num_elements,
                                mpi_ctx_.GetMPIDataType(first_entry.output),
                                first_entry.recv_neighbors->at(i),
                                /*tag=*/mpi_ctx_.rank_ + first_entry.recv_neighbors->at(i),
                                mpi_ctx_.GetMPICommunicator(Communicator::GRAPH),
                                &requests[i + nsend]);
        if (ret_code != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI_Irecv (for dynamic neighbor_allreduce) failed, see MPI output "
              "for details.");
        }
      }
      for (int i = 0; i < nsend; ++i) {
        int ret_code = MPI_Isend(
            fused_input_data, num_elements, mpi_ctx_.GetMPIDataType(first_entry.tensor),
            first_entry.send_neighbors->at(i),
            /*tag=*/mpi_ctx_.rank_ + first_entry.send_neighbors->at(i),
            mpi_ctx_.GetMPICommunicator(Communicator::GRAPH), &requests[i]);
        if (ret_code != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI_Isend (for dynamic neighbor_allreduce) failed, see MPI output "
              "for details.");
        }
      }
      MPI_Waitall(nsend + nrecv, requests.data(), statuses.data());
      error_message =
          GenerateNeighborAllreduceErrorMessage(statuses, nsend, nrecv);
    } else {
      // Do nothing here.
    }
    // Because the in-place modification, we need to copy fused_input_data back to tensor as well
    MemcpyOutFusionBufferForInputs(fused_input_data, entries);
    // 3. Broadcast recv data from local rank = 0 to other local ranks.
    int recv_num_elements = num_elements * first_entry.recv_neighbors->size();
    MPI_Bcast(buffer_data, recv_num_elements,
              mpi_ctx_.GetMPIDataType(first_entry.output), 0,
              mpi_ctx_.GetMPICommunicator(Communicator::LOCAL));
  }
  timeline_ptr->ActivityEndAll(entries);

  // Remember buffer_data is already pointed at offset location (after self tensor).
  timeline_ptr->ActivityStartAll(entries, "MEMCPY_OUT_FUSION_BUFFER");
  int num_recv_neighbors = !first_entry.dynamic_neighbors_enabled
                           ? mpi_ctx_.neighbor_indgree_
                           : first_entry.recv_neighbors->size();
  MemcpyOutFusionBufferForNeighbors(
      buffer_data, entries, num_recv_neighbors,
      /*fused_data_size=*/ num_elements * element_size);
  timeline_ptr->ActivityEndAll(entries);

  for (auto& e : entries) {
    if (error_message != "") {
      e.callback(Status::UnknownError(error_message));
    } else {
      e.callback(Status::OK());
    }
  }
}

void MPIController::PairGossip(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  void* recvbuf = (void*)entry.output->data();
  const int num_elements = entry.tensor->shape().num_elements();
  const int recv_num_elements = entry.output->shape().num_elements();
  const int target_rank = entry.root_rank;  // We re-use root rank (broadcast)
                                            // as target rank in pair gossip.

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);
  with_device device_guard(entry.device);

  timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
  int ret_code = MPI_Sendrecv(
      sendbuf, num_elements, mpi_ctx_.GetMPIDataType(entry.tensor), target_rank,
      0, recvbuf, recv_num_elements, mpi_ctx_.GetMPIDataType(entry.output),
      target_rank, 0, mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL),
      MPI_STATUS_IGNORE);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "Pair_gossip(through MPI_Sendrecv) failed, see MPI output for "
        "details.");
  }
  timeline_ptr->ActivityEnd(entry.tensor_name);

  entry.callback(Status::OK());
}

bool MPIController::IsMpiUnifiedModel() {
  void* data_buf = nullptr;
  int win_size = 1;
  int element_size = 1;
  MPI_Win fake_win;
  MPI_Win_create(data_buf, win_size, element_size, MPI_INFO_NULL,
                 mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL), &fake_win);
  int flag = 0;
  int* memory_model;
  MPI_Win_get_attr(fake_win, MPI_WIN_MODEL, &memory_model, &flag);
  MPI_Win_free(&fake_win);
  if (flag == 0) {
    BFLOG(WARNING) << "Failed to get MPI_WIN_MODEL attribution";
    return false;
  }
  BFLOG(DEBUG) << "Unified MPI_WIN_MODEL support is "
               << (*memory_model == MPI_WIN_UNIFIED);
  return *memory_model == MPI_WIN_UNIFIED;
}

void MPIController::WinCreate(TensorTableEntry& entry) {
  with_device device_guard(entry.device);

  std::shared_ptr<Tensor>& tensor = entry.tensor;
  std::vector<std::shared_ptr<Tensor>>& neighbor_tensors = entry.neighbor_tensors;
  const std::string& name = entry.tensor_name;
  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  timeline_ptr->ActivityStart(name, "WIN_CREATE");
  // We need to explicitly set the device here.
  // 1. Regist a Name and create a window first.
  // It also initializes the mutex memoery.
  if (!mpi_ctx_.RegisterWindowName(name)) {
    entry.callback(
        Status::InvalidArgument(std::string("Win_create failed with ") + name));
    return;
  }
  // 2. Get the registered window manager.
  std::shared_ptr<WindowManager> win_manager = mpi_ctx_.GetWindowByName(name);

  // A global win hold the self memory, used by win_accumulate and win_get.
  auto global_mpi_win_ptr = std::make_shared<MPI_Win>();
  void* data_buf = (void*)tensor->data();
  int element_size = mpi_ctx_.GetMPITypeSize(tensor->dtype());
  int win_size = (tensor->shape().num_elements()) * element_size;
  MPI_Win_create(data_buf, win_size, element_size, MPI_INFO_NULL,
                 mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL),
                 global_mpi_win_ptr.get());
  win_manager->SetGlobalWin(global_mpi_win_ptr);

  // Build extra buffers for win_put.
  // For example: size=4 exponential two ring topology
  // r\s   0    1    2    3
  //  0    g    x         x
  //  1    x    g    x
  //  2         x    g    x
  //  3    x         x    g
  //  The following for-loop scans along columns
  //  and the self-rank determines the rows.
  //  If there is connection, the window is associated with the
  //  neighbor_tensors. Otherwise, window is associated with null pointer.
  std::shared_ptr<MPI_Win> mpi_win_ptr;
  int neighbor_tensor_index = 0;
  for (int rank = 0; rank < mpi_ctx_.size_; rank++) {
    auto mpi_win_ptr = std::make_shared<MPI_Win>();
    std::shared_ptr<Tensor> t = nullptr;
    if (rank == mpi_ctx_.rank_) {
      // Sender (no need to allocate the memory with it.)
      data_buf = nullptr;
      element_size = 1;
      win_size = 0;
    } else if (std::find(mpi_ctx_.neighbor_in_ranks_.begin(), mpi_ctx_.neighbor_in_ranks_.end(),
                         rank) != mpi_ctx_.neighbor_in_ranks_.end()) {
      // Receiver
      t = neighbor_tensors[neighbor_tensor_index++];
      data_buf = (void*)t->data();
      element_size = mpi_ctx_.GetMPITypeSize(t->dtype());
      win_size = (t->shape().num_elements()) * element_size;
    } else {
      // Just participate in a collective call.
      data_buf = nullptr;
      element_size = 1;
      win_size = 0;
    }
    MPI_Win_create(data_buf, win_size, element_size, MPI_INFO_NULL,
                   mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL),
                   mpi_win_ptr.get());
    win_manager->PushBackWinAndTensor(mpi_win_ptr, t);
  }
  timeline_ptr->ActivityEnd(name);

  entry.callback(Status::OK());
}

void MPIController::WinFree(TensorTableEntry& entry) {
  if (!mpi_ctx_.UnregisterWindowName(entry.tensor_name)) {
    entry.callback(Status::InvalidArgument(
        std::string("Win_free failed with ") + entry.tensor_name));
    return;
  }
  entry.callback(Status::OK());
}

void MPIController::WinFreeAll(TensorTableEntry& entry) {
  if (!mpi_ctx_.UnregisterAllWindowName()) {
    entry.callback(
        Status::InvalidArgument(std::string("Win_free_all failed.")));
    return;
  }
  BFLOG(DEBUG) << "All MPI Win has been freed.";
  entry.callback(Status::OK());
}

Status MPIController::WinSync(const std::string& name, int device, bool with_associated_p) {
  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::InvalidArgument(std::string("Win_sync failed with ") + name);
  }

  with_device device_guard(device);
  auto win_mananger = it->second;
  for (auto rank : mpi_ctx_.neighbor_in_ranks_) {
    auto mpi_win_ptr = win_mananger->GetWinByRank(rank);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, mpi_ctx_.rank_, MPI_MODE_NOCHECK, *mpi_win_ptr);
    MPI_Win_sync(*mpi_win_ptr);
    MPI_Win_unlock(mpi_ctx_.rank_, *mpi_win_ptr);
  }
  if (with_associated_p) {
    auto p_win_ptr = win_mananger->GetPWin();
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, mpi_ctx_.rank_, MPI_MODE_NOCHECK,
                 *p_win_ptr);
    MPI_Win_sync(*p_win_ptr);
    MPI_Win_unlock(mpi_ctx_.rank_, *p_win_ptr);
  }

  VersionWinClear(name);

  return Status::OK();
}

Status MPIController::WinFence(const std::string& name) {
  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::InvalidArgument(std::string("Win_fence failed with ") + name);
  }

  std::shared_ptr<WindowManager> win_mananger = it->second;
  for (int rank = 0; rank < mpi_ctx_.size_; rank++) {
    MPI_Win_fence(0, *(win_mananger->GetWinByRank(rank)));
  }

  return Status::OK();
}

// Reshuffle the order of destination to avoid the collision of network.
std::vector<std::pair<int, double>> GetSortedDstWeights(
    const int self_rank, const int size, const std::unordered_map<int, double> dst_weights) {
  std::vector<std::pair<int, double>> sorted_dst_weights;
  for (auto kv : dst_weights) {
    int target_rank = kv.first;
    double weight = kv.second;
    sorted_dst_weights.push_back(std::make_pair(target_rank, weight));
  }

  std::sort(
      sorted_dst_weights.begin(), sorted_dst_weights.end(),
      [self_rank, size](std::pair<int, double> a, std::pair<int, double> b) {
        int distance1 = a.first - self_rank;
        int distance2 = b.first - self_rank;
        if (a.first < self_rank) distance1 += size;
        if (b.first < self_rank) distance2 += size;
        return distance1 < distance2;
      });
  return sorted_dst_weights;
}

void MPIController::WinPut(TensorTableEntry& entry) {
  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  int num_elements = entry.tensor->shape().num_elements();
  MPI_Datatype data_type = mpi_ctx_.GetMPIDataType(entry.tensor);
  auto it = mpi_ctx_.named_win_map.find(entry.tensor_name);
  if (it == mpi_ctx_.named_win_map.end()) {
    throw std::runtime_error(std::string("Cannot find ") + entry.tensor_name +
                             " in (MPI) registered win name.");
  }
  std::shared_ptr<WindowManager> win_mananger = it->second;
  MPI_Win mpi_win = *(win_mananger->GetWinByRank(mpi_ctx_.rank_));

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  std::vector<std::pair<int, double>> sorted_dst_weights =
      GetSortedDstWeights(mpi_ctx_.rank_, mpi_ctx_.size_, entry.dst_weights);

  for (auto kv : sorted_dst_weights) {
    int target_rank = kv.first;
    double weight = kv.second;

    BFLOG(TRACE, mpi_ctx_.rank_) << "Start MPI_Put for " << entry.tensor_name << " to " << target_rank;

    if (entry.require_mutex) {
      timeline_ptr->ActivityStart(entry.tensor_name, "Aquire_Mutex");
      WinMutexAcquire(entry.tensor_name, {target_rank}, /*is_sync=*/false);
      timeline_ptr->ActivityEnd(entry.tensor_name);
    }
    timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, MPI_MODE_NOCHECK, mpi_win);
    // avoid putting the tensor for itself (NOT valid).
    if (target_rank == mpi_ctx_.rank_) continue;
    auto tensor = entry.tensor->data_weight(weight);
    void* sendbuf = (void*)tensor->data();
    int target_disp = 0;  // offset in win buffer
    int sent_size = std::min(MAX_WIN_SENT, num_elements - target_disp);
    while (sent_size != 0) {
      void* sendbuf_start =
          (void*)(static_cast<char*>(sendbuf) +
                  target_disp * mpi_ctx_.GetMPITypeSize(tensor->dtype()));
      int ret_code = MPI_Put(sendbuf_start, sent_size, data_type, target_rank,
                             target_disp, sent_size, data_type, mpi_win);
      if (ret_code != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Put failed, see MPI output for details.");
      }
      target_disp += sent_size;
      sent_size = std::min(MAX_WIN_SENT, num_elements - target_disp);
    }
    MPI_Win_unlock(target_rank, mpi_win);
    timeline_ptr->ActivityEnd(entry.tensor_name);

    WinVersionPutUpdate(entry.tensor_name, {target_rank});

    if (entry.win_ops_with_associated_p) {
      std::shared_ptr<MPI_Win> weight_win = win_mananger->GetPWin();
      MPI_Win_lock(MPI_LOCK_SHARED, target_rank, MPI_MODE_NOCHECK, *weight_win);
      // Unlike data window, weight window is just a raw "world size" vector.
      int target_disp = mpi_ctx_.rank_;
      double* p_memory = win_mananger->GetUnderlyingPMemory();
      double weighted_p = (*(p_memory + mpi_ctx_.rank_)) * weight;
      int ret_code = MPI_Put(&weighted_p, 1, MPI_DOUBLE, target_rank,
                             target_disp, 1, MPI_DOUBLE, *weight_win);
      if (ret_code != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Put failed, see MPI output for details.");
      }
      MPI_Win_unlock(target_rank, *weight_win);
    }

    if (entry.require_mutex) {
      WinMutexRelease(entry.tensor_name, {target_rank}, /*is_sync=*/false);
    }
  }

  BFLOG(TRACE, mpi_ctx_.rank_) << "MPI_Put for " << entry.tensor_name << " is done.";

  timeline_ptr->ActivityStart(entry.tensor_name, "CALLBACK");
  entry.callback(Status::OK());
  timeline_ptr->ActivityEnd(entry.tensor_name);
}

void MPIController::WinAccumulate(TensorTableEntry& entry) {
  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  int num_elements = entry.tensor->shape().num_elements();
  MPI_Datatype data_type = mpi_ctx_.GetMPIDataType(entry.tensor);
  auto it = mpi_ctx_.named_win_map.find(entry.tensor_name);
  if (it == mpi_ctx_.named_win_map.end()) {
    throw std::runtime_error(std::string("Cannot find ") + entry.tensor_name +
                             " in (MPI) registered win name.");
  }
  std::shared_ptr<WindowManager> win_mananger = it->second;
  MPI_Win mpi_win = *(win_mananger->GetWinByRank(mpi_ctx_.rank_));

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  std::vector<std::pair<int, double>> sorted_dst_weights =
      GetSortedDstWeights(mpi_ctx_.rank_, mpi_ctx_.size_, entry.dst_weights);

  for (auto kv : sorted_dst_weights) {
    int target_rank = kv.first;
    double weight = kv.second;
    // avoid putting the tensor for itself (NOT valid).
    if (target_rank == mpi_ctx_.rank_) continue;

    if (entry.require_mutex) {
      timeline_ptr->ActivityStart(entry.tensor_name, "Aquire_Mutex");
      WinMutexAcquire(entry.tensor_name, {target_rank}, /*is_sync=*/false);
      timeline_ptr->ActivityEnd(entry.tensor_name);
    }
    auto tensor = entry.tensor->data_weight(weight);
    void* sendbuf = (void*)tensor->data();

    timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");

    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, MPI_MODE_NOCHECK, mpi_win);
    int target_disp = 0;  // offset in win buffer
    int sent_size = std::min(MAX_WIN_SENT, num_elements - target_disp);
    while (sent_size != 0) {
      void* sendbuf_start =
          (void*)(static_cast<char*>(sendbuf) +
                  target_disp * mpi_ctx_.GetMPITypeSize(tensor->dtype()));
      int ret_code =
          MPI_Accumulate(sendbuf_start, sent_size, data_type, target_rank,
                         target_disp, sent_size, data_type, MPI_SUM, mpi_win);
      if (ret_code != MPI_SUCCESS) {
        if (entry.require_mutex)
          WinMutexRelease(entry.tensor_name, {target_rank}, /*is_sync=*/false);
        throw std::runtime_error(
            "MPI_Accumulate failed, see MPI output for details.");
      }
      target_disp += sent_size;
      sent_size = std::min(MAX_WIN_SENT, num_elements - target_disp);
    }
    MPI_Win_unlock(target_rank, mpi_win);
    timeline_ptr->ActivityEnd(entry.tensor_name);

    if (entry.win_ops_with_associated_p) {
      std::shared_ptr<MPI_Win> weight_win = win_mananger->GetPWin();
      MPI_Win_lock(MPI_LOCK_SHARED, target_rank, MPI_MODE_NOCHECK, *weight_win);
      // Unlike data window, weight window is just a raw "world size" vector.
      int target_disp = mpi_ctx_.rank_;
      double* p_memory = win_mananger->GetUnderlyingPMemory();
      double weighted_p = (*(p_memory + mpi_ctx_.rank_)) * weight;
      int ret_code =
          MPI_Accumulate(&weighted_p, 1, MPI_DOUBLE, target_rank, target_disp,
                         1, MPI_DOUBLE, MPI_SUM, *weight_win);
      if (ret_code != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Accumulate failed, see MPI output for details.");
      }
      MPI_Win_unlock(target_rank, *weight_win);
    }

    if (entry.require_mutex) {
      WinMutexRelease(entry.tensor_name, {target_rank}, /*is_sync=*/false);
    }
  }
  BFLOG(TRACE, mpi_ctx_.rank_)
      << "MPI_Accmulate for " << entry.tensor_name << " is done.";

  timeline_ptr->ActivityStart(entry.tensor_name, "CALLBACK");
  entry.callback(Status::OK());
  timeline_ptr->ActivityEnd(entry.tensor_name);
}

void MPIController::WinGet(TensorTableEntry& entry) {
  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  auto it = mpi_ctx_.named_win_map.find(entry.tensor_name);
  if (it == mpi_ctx_.named_win_map.end()) {
    throw std::runtime_error(std::string("Cannot find ") + entry.tensor_name +
                             std::string(" in (MPI) registered win object name."));
  }
  std::shared_ptr<WindowManager> win_mananger = it->second;
  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  MPI_Win mpi_win = *(win_mananger->GetGlobalWin());
  for (auto kv : entry.src_weights) {
    int target_rank = kv.first;
    // avoid getting the tensor for itself.
    if (target_rank == mpi_ctx_.rank_) continue;

    if (entry.require_mutex) {
      timeline_ptr->ActivityStart(entry.tensor_name, "Aquire_Mutex");
      WinMutexAcquire(entry.tensor_name, {target_rank}, /*is_sync=*/false);
      timeline_ptr->ActivityEnd(entry.tensor_name);
    }

    auto tensor = win_mananger->GetAssociateTensorByRank(target_rank);
    void* recvbuf = (void*)tensor->data();
    int num_elements = tensor->shape().num_elements();
    MPI_Datatype data_type = mpi_ctx_.GetMPIDataType(tensor);

    BFLOG(DEBUG, mpi_ctx_.rank_) << "MPI_Get for " << entry.tensor_name << " is to get "
                        << num_elements << " from " << target_rank;

    timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, MPI_MODE_NOCHECK, mpi_win);
    int target_disp = 0;  // offset in win buffer
    int recv_size = std::min(MAX_WIN_SENT, num_elements - target_disp);
    while (recv_size != 0) {
      void* recvbuf_start =
          (void*)(static_cast<char*>(recvbuf) +
                  target_disp * mpi_ctx_.GetMPITypeSize(tensor->dtype()));
      int ret_code = MPI_Get(recvbuf_start, recv_size, data_type, target_rank,
                             target_disp, recv_size, data_type, mpi_win);
      if (ret_code != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Get failed, see MPI output for details.");
      }
      target_disp += recv_size;
      recv_size = std::min(MAX_WIN_SENT, num_elements - target_disp);
    }
    MPI_Win_unlock(target_rank, mpi_win);
    timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");

    WinVersionGetUpdate(entry.tensor_name, {target_rank});

    if (entry.require_mutex) {
      WinMutexRelease(entry.tensor_name, {target_rank}, /*is_sync=*/false);
    }
  }

  BFLOG(TRACE, mpi_ctx_.rank_) << "Win_get for " << entry.tensor_name << " is done.";
  entry.callback(Status::OK());
}

void MPIController::Barrier(TensorTableEntry& entry) {
  int ret_code = MPI_Barrier(mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL));
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Barrier failed, see MPI output for details.");
  }
  entry.callback(Status::OK());
}

Status MPIController::WinLock(const std::string& name) {
  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::InvalidArgument(
        std::string("Cannot find ") + name +
        std::string(" in registered win object name."));
  }
  std::shared_ptr<WindowManager> win_mananger = it->second;
  MPI_Win mpi_win = *(win_mananger->GetGlobalWin());

  // It only locks the memory in local.
  int target_rank = mpi_ctx_.rank_;
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, MPI_MODE_NOCHECK, mpi_win);

  for (const int& rank : mpi_ctx_.neighbor_in_ranks_) {
    auto mpi_win_ptr = win_mananger->GetWinByRank(rank);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, MPI_MODE_NOCHECK,
                 *mpi_win_ptr);
  }

  return Status::OK();
}

Status MPIController::WinUnlock(const std::string& name) {
  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::InvalidArgument(
        std::string("Cannot find ") + name +
        std::string(" in registered win object name."));
  }
  std::shared_ptr<WindowManager> win_mananger = it->second;
  MPI_Win mpi_win = *(win_mananger->GetGlobalWin());

  // It only locks the memory in local.
  int target_rank = mpi_ctx_.rank_;
  MPI_Win_unlock(target_rank, mpi_win);

  for (const int& rank : mpi_ctx_.neighbor_in_ranks_) {
    auto mpi_win_ptr = win_mananger->GetWinByRank(rank);
    MPI_Win_unlock(target_rank, *mpi_win_ptr);
  }

  return Status::OK();
}

Status MPIController::WinMutexAcquire(const std::string& name,
                                      const std::vector<int>& acquire_ranks,
                                      bool is_sync) {
  BFLOG(TRACE, mpi_ctx_.rank_) << "Win Mutex for " << name << " is acquired.";
  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::PreconditionError(
        "Cannot accquire Mutex Win for " + name +
        ". It may not be created or has "
        "been destroyed or wrong name for associated window.");
  }
  std::shared_ptr<MPI_Win> mutex_win = it->second->GetMutexWin();
  if (!mutex_win) {
    return Status::PreconditionError("Cannot accuire Mutex Win for " + name +
                                     ". The data window for that name is found"
                                     "but the mutex window is not.");
  }
  return MPIWinMutexAcquireImpl(mutex_win, acquire_ranks, mpi_ctx_.rank_, is_sync);
}

Status MPIController::WinMutexRelease(const std::string& name,
                                      const std::vector<int>& release_ranks,
                                      bool is_sync) {
  BFLOG(TRACE, mpi_ctx_.rank_) << "Win Mutex for " << name << " is released.";

  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::PreconditionError(
        "Cannot release Mutex Win for " + name +
        ". It may not be created or has "
        "been destroyed or wrong name for associated window.");
  }
  std::shared_ptr<MPI_Win> mutex_win = it->second->GetMutexWin();
  if (!mutex_win) {
    return Status::PreconditionError("Cannot release Mutex Win for " + name +
                                     ". The data window for that name is found"
                                     "but mutex window is not.");
  }
  return MPIWinMutexReleaseImpl(mutex_win, release_ranks, mpi_ctx_.rank_, is_sync);
}


/**
 * This function increaments the local version for the corresponding rank
 * when there is a win get operation.
 **/
Status MPIController::WinVersionGetUpdate(const std::string& name,
                                          const std::vector<int>& ranks) {
  BFLOG(TRACE, mpi_ctx_.rank_) << "Update Win Version for " << name << ".";

  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::PreconditionError(
        "Cannot get Version Win for " + name +
        ". It may not be created or has "
        "been destroyed or wrong name for associated window.");
  }
  std::shared_ptr<MPI_Win> version_win = it->second->GetVersionWin();
  if (!version_win) {
    return Status::PreconditionError("Cannot get Version Win for " + name +
                                     ". The data window for that name is found"
                                     "but the version window is not.");
  }

  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, mpi_ctx_.rank_, MPI_MODE_NOCHECK,
               *version_win);
  for (int position : ranks) {
    it->second->incrementVersionWinMem(position);
  }
  MPI_Win_sync(*version_win);
  MPI_Win_unlock(mpi_ctx_.rank_, *version_win);

  return Status::OK();
}

/**
 * This function increaments the remote version for the corresponding rank
 * when there is a win put operation.
 **/
Status MPIController::WinVersionPutUpdate(const std::string& name,
                                          const std::vector<int>& ranks) {
  BFLOG(TRACE, mpi_ctx_.rank_) << "Update Win Version for " << name << ".";

  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::PreconditionError(
        "Cannot accquire Version Win for " + name +
        ". It may not be created or has "
        "been destroyed or wrong name for associated window.");
  }
  std::shared_ptr<MPI_Win> version_win = it->second->GetVersionWin();
  if (!version_win) {
    return Status::PreconditionError("Cannot accuire Version Win for " + name +
                                     ". The data window for that name is found"
                                     "but the version window is not.");
  }

  int one = 1;

  for (int rank : ranks) {
    MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, *version_win);
    MPI_Accumulate(&one, 1, MPI_INT, rank, /*target_disp=*/mpi_ctx_.rank_, 1,
                   MPI_INT, MPI_SUM, *version_win);
    MPI_Win_unlock(rank, *version_win);
  }

  return Status::OK();
}

Status MPIController::VersionWinClear(const std::string& name) {
  BFLOG(TRACE, mpi_ctx_.rank_) << "Win Version for " << name << " is released.";

  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::PreconditionError(
        "Cannot clear Version Win for " + name +
        ". It may not be created or has "
        "been destroyed or wrong name for associated window.");
  }
  std::shared_ptr<MPI_Win> version_win = it->second->GetVersionWin();
  if (!version_win) {
    return Status::PreconditionError("Cannot release Version Win for " + name +
                                     ". The data window for that name is found"
                                     "but version window is not.");
  }

  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, mpi_ctx_.rank_, MPI_MODE_NOCHECK,
               *version_win);
  it->second->resetVersionWinMem();
  MPI_Win_sync(*version_win);
  MPI_Win_unlock(mpi_ctx_.rank_, *version_win);

  return Status::OK();
}

Status MPIController::GetWindowVersionValue(const std::string& name,
                                            std::vector<int>& versions) {
  BFLOG(TRACE, mpi_ctx_.rank_)
      << "Get Win Version for " << name << " is released.";

  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::PreconditionError(
        "Cannot get Version Win for " + name +
        ". It may not be created or has "
        "been destroyed or wrong name for associated window.");
  }
  std::shared_ptr<MPI_Win> version_win = it->second->GetVersionWin();
  std::vector<int> version_mem = it->second->GetVersionMemoryCopy();
  for (int i = 0; i < version_mem.size(); i++) {
    versions[i] = version_mem[i];
  }

  return Status::OK();
}

void MPIController::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, void*& buffer_data,
    size_t& buffer_len) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  FusionBufferManager* buffer_manager;
  auto fusion_status = GetBluefogFusionBuffer(buffer_manager);
  if (!fusion_status.ok()){
    throw std::runtime_error(fusion_status.reason());
  }
  std::shared_ptr<PersistentBuffer> buffer =
      buffer_manager->GetBuffer(first_entry.device);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryInFusionBuffer(e, buffer_data_at_offset);
    offset += e.tensor->size();
  }

  buffer_len = (size_t)offset;
}

void MPIController::MemcpyOutFusionBuffer(
    const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryOutFusionBuffer(buffer_data_at_offset, e);
    offset += e.output->size();
  }
}

void MPIController::MemcpyOutFusionBufferForNeighbors(
    const void* buffer_data, std::vector<TensorTableEntry>& entries,
    const int num_recv_neighbors, const int64_t fused_data_size) {
  // Remember buffer_data is already pointed at offset location (after self).
  // Unfornately, we cannot simply use MemcpyOutFusionBuffer because:
  // buffer -- [t_1, t_2 | t_1_n1, t_2_n1, t_1_n2, t_2_n2]
  // needs to split into [t_1_n1, t_1_n2] and [t_2_n1, t_2_n2].
  // Notice the size of t_1_n1 can be retrieved from the tensor size.
  // And the size of [t_1_n1, t_1_n2] can be retrieved from the output size.
  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryOutFusionBufferForNeighbors(buffer_data_at_offset, e,
                                           num_recv_neighbors, fused_data_size);
    offset += e.tensor->size();
  }
}

void MPIController::MemcpyOutFusionBufferForInputs(
    const void* fused_input_data, std::vector<TensorTableEntry>& entries) {
  // Copy the input data stored in the fusion buffer back to input, which is
  // used in hierarchical neighbor allreduce since it has allreduce step to
  // modified the input data.
  int64_t offset = 0;
  for (auto& e : entries) {
    void* fused_input_data_at_offset = (uint8_t*)fused_input_data + offset;
    void* dst_data = (void*)e.tensor->data();
    size_t count = (size_t)e.tensor->size();
#if HAVE_CUDA
    if (e.device != CPU_DEVICE_ID) {
      CUDACHECK(cudaMemcpy(dst_data, fused_input_data_at_offset, count,
                           cudaMemcpyDeviceToDevice));
    } else {
#endif
      std::memcpy(dst_data, fused_input_data, count);
#if HAVE_CUDA
    }
#endif
    offset += e.tensor->size();
  }
}

void MPIController::MemcpyEntryInFusionBuffer(const TensorTableEntry& e,
                                              void* buffer_data_at_offset) {
  const void* src_data = e.tensor->data();
  size_t count = (size_t)e.tensor->size();
#if HAVE_CUDA
  if (e.device != CPU_DEVICE_ID) {
    CUDACHECK(cudaMemcpy(buffer_data_at_offset, src_data, count,
                         cudaMemcpyDeviceToDevice));
  } else {
#endif
    std::memcpy(buffer_data_at_offset, src_data, count);
#if HAVE_CUDA
  }
#endif
}

void MPIController::MemcpyEntryOutFusionBuffer(
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  void* dst_data = (void*)e.output->data();
  size_t count = (size_t)e.output->size();
#if HAVE_CUDA
  if (e.device != CPU_DEVICE_ID) {
    CUDACHECK(cudaMemcpy(dst_data, buffer_data_at_offset, count,
                         cudaMemcpyDeviceToDevice));
  } else {
#endif
    std::memcpy(dst_data, buffer_data_at_offset, count);
#if HAVE_CUDA
  }
#endif
}

void MPIController::MemcpyEntryOutFusionBufferForNeighbors(
    const void* buffer_data_at_offset, TensorTableEntry& e,
    const int num_recv_neighbors, const int64_t fused_data_size) {
  // The buffer data looks like
  // [t_1, t_2 | t_1_n1, t_2_n1, t_1_n2, t_2_n2]
  //           ^               ^
  //           |-------------->| fused_data_size
  //           buffer_data_at_offset
  // Output for t_1 is [t_1_n1, t_1_n2]
  //            t_2 is [t_2_n1, t_2_n2]
  for (int i = 0; i < num_recv_neighbors; ++i) {
    void* output_at_offset =
        (uint8_t*)e.output->data() + i * (size_t)e.tensor->size();
    void* buffer_data_at_offset_for_neighbor =
        (uint8_t*)buffer_data_at_offset + i * fused_data_size;
    size_t count = (size_t)e.tensor->size();
#if HAVE_CUDA
    if (e.device != CPU_DEVICE_ID) {
      CUDACHECK(cudaMemcpy(output_at_offset, buffer_data_at_offset_for_neighbor,
                           count, cudaMemcpyDeviceToDevice));
    } else {
#endif
      std::memcpy(output_at_offset, buffer_data_at_offset_for_neighbor, count);
#if HAVE_CUDA
    }
#endif
  }
}

// Extracted from book "Using Advanced MPI" Section 4.5
Status MPIWinMutexAcquireImpl(std::shared_ptr<MPI_Win> mutex_win,
                              const std::vector<int>& acquire_ranks,
                              int self_rank, bool is_sync) {
  // TODO(ybc) Try better implementation than Spin Lock.
  // Recall that we build N windows across all N processes.
  // The spin value is stored in the rank i for i-th window.
  // Other process will got to acquire it.
  int one = 1;
  int minus_one = -1;
  int oldval = 0;

  if (is_sync) {  // Lock for self mutex
    MPI_Win_lock(MPI_LOCK_SHARED, self_rank, 0, *mutex_win);
    for (int rank : acquire_ranks) {
      do {
        MPI_Fetch_and_op(&one, &oldval, MPI_INT, self_rank,
                         /*target_disp=*/rank, MPI_SUM, *mutex_win);
        MPI_Win_flush(self_rank, *mutex_win);
        if (oldval == 0) break;
        MPI_Accumulate(&minus_one, 1, MPI_INT, self_rank, /*target_disp=*/rank,
                       1, MPI_INT, MPI_SUM, *mutex_win);
        MPI_Win_flush(self_rank, *mutex_win);
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      } while (1);
    }
    MPI_Win_unlock(self_rank, *mutex_win);
  } else {  // Lock for remote mutex
    for (int rank : acquire_ranks) {
      MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, *mutex_win);
      do {
        MPI_Fetch_and_op(&one, &oldval, MPI_INT, rank,
                         /*target_disp=*/self_rank, MPI_SUM, *mutex_win);
        MPI_Win_flush(rank, *mutex_win);
        if (oldval == 0) break;
        MPI_Accumulate(&minus_one, 1, MPI_INT, rank,
                       /*target_disp=*/self_rank, 1, MPI_INT, MPI_SUM,
                       *mutex_win);
        MPI_Win_flush(rank, *mutex_win);
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      } while (1);
      MPI_Win_unlock(rank, *mutex_win);
    }
  }

  return Status::OK();
}

Status MPIWinMutexReleaseImpl(std::shared_ptr<MPI_Win> mutex_win,
                              const std::vector<int>& release_ranks,
                              int self_rank, bool is_sync) {
  int minus_one = -1;
  // TODO(ybc) Notice the following accumulate may cause the value to be
  // negative, i.e. more release ops is called than acquire.
  if (is_sync) {
    MPI_Win_lock(MPI_LOCK_SHARED, self_rank, 0, *mutex_win);
    for (int rank : release_ranks) {
      MPI_Accumulate(&minus_one, 1, MPI_INT, self_rank, /*target_disp=*/rank, 1,
                     MPI_INT, MPI_SUM, *mutex_win);
    }
    MPI_Win_unlock(self_rank, *mutex_win);
  } else {
    for (int rank : release_ranks) {
      MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, *mutex_win);
      MPI_Accumulate(&minus_one, 1, MPI_INT, rank, /*target_disp=*/self_rank, 1,
                     MPI_INT, MPI_SUM, *mutex_win);
      MPI_Win_unlock(rank, *mutex_win);
    }
  }
  return Status::OK();
}

Status MPIController::GetWinAssociatedPByNameAndRank(const std::string& name,
                                                     const int rank,
                                                     double* weight) {
  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::PreconditionError(
        "Cannot get win associated P for " + name +
        ". It may not be created or has been destroyed or wrong name for "
        "associated window.");
  }
  if (rank < 0 || rank >= mpi_ctx_.size_) {
    return Status::PreconditionError(
        "Argument Rank to retrieve win associated P should be a value between "
        "0 (inclusive) and size(exclusive).");
  }
  *weight = it->second->GetAssociatedP(rank);
  return Status::OK();
}

Status MPIController::SetWinAssociatedPByNameAndRank(const std::string& name,
                                                     const int rank,
                                                     double weight) {
  auto it = mpi_ctx_.named_win_map.find(name);
  if (it == mpi_ctx_.named_win_map.end()) {
    return Status::PreconditionError(
        "Cannot get win associated P for " + name +
        ". It may not be created or has been destroyed or wrong name for "
        "associated window.");
  }
  if (rank < 0 || rank >= mpi_ctx_.size_) {
    return Status::PreconditionError(
        "Argument Rank to retrieve associated P should be a value "
        "between 0 (inclusive) and size(exclusive).");
  }
  it->second->SetAssociatedP(rank, weight);
  return Status::OK();
}

}  // namespace common
}  // namespace bluefog
