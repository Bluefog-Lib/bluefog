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

#include "nccl_controller.h"

#include <algorithm>
#include <string>
#include <thread>

#include "common.h"
#include "cuda_util.h"
#include "mpi_controller.h"
#include "operations.h"
#include "timeline.h"

namespace bluefog {
namespace common {

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
    case DataType::BLUEFOG_INT32:
      return ncclInt32;
    case DataType::BLUEFOG_INT64:
      return ncclInt64;
    case DataType::BLUEFOG_FLOAT16:
      return ncclFloat16;
    case DataType::BLUEFOG_FLOAT32:
      return ncclFloat32;
    case DataType::BLUEFOG_FLOAT64:
      return ncclFloat64;
    default:
      throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                             " is not supported in NCCL mode.");
  }
}

void NCCLContext::Initialize(const int rank, const int size,
                             const int local_rank) {
  if (is_initialized) {
    BFLOG(DEBUG)
        << "NCCL context has been initialized but NCCLContext::Initialize "
        << "is called again";
    return;
  }

  // A single rank will create a unique ID and send it to all other ranks to
  // make sure everyone has it.
  ncclUniqueId nccl_id;
  if (rank == 0) ncclGetUniqueId(&nccl_id);
  MPICHECK(
      MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Assume one device per process
  int nDevices = 0;
  CUDACHECK(cudaGetDeviceCount(&nDevices));
  CUDACHECK(cudaSetDevice(local_rank % nDevices));
  int greatest_priority;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
  CUDACHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                         greatest_priority));
  NCCLCHECK(ncclCommInitRank(&nccl_comm, size, nccl_id, rank));

  is_initialized = true;
  cuda_device = local_rank % nDevices;
  return;
}

#if NCCL_MINOR < 7
void NCCLContext::CleanPeerCommunicator() {
  for (const auto& pair : pair_order) {
    BFLOG(DEBUG) << "Destory comm for pair (" << pair.first << ", "
                 << pair.second << ")";
    ncclComm_t& pair_nccl_comm = nccl_pair_comms.at(pair);
    NCCLCHECK(ncclCommDestroy(pair_nccl_comm));
    cudaStream_t& pair_stream = pair_streams.at(pair);
    CUDACHECK(cudaStreamDestroy(pair_stream));
  }
  pair_order.clear();
  pair_streams.clear();
  nccl_pair_comms.clear();
}
#endif

void NCCLContext::Finalize() {
  NCCLCHECK(ncclCommDestroy(nccl_comm));
  CUDACHECK(cudaStreamDestroy(stream));

#if NCCL_MINOR < 7
  CleanPeerCommunicator();
#endif

  is_initialized = false;
  cuda_device = -1;
}

cudaError_t NCCLContext::GetCudaEvent(cudaEvent_t* event) {
  auto& mutex = cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    if (!cuda_events.empty()) {
      *event = cuda_events.front();
      cuda_events.pop();
      return cudaSuccess;
    }
  }

  return cudaEventCreateWithFlags(
      event, cudaEventBlockingSync | cudaEventDisableTiming);
}

cudaError_t NCCLContext::ReleaseCudaEvent(cudaEvent_t event) {
  auto& mutex = cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    cuda_events.push(event);
  }

  return cudaSuccess;
}

void NCCLController::Initialize() {
  nccl_ctx_.Initialize(mpi_ctx_.rank_, mpi_ctx_.size_, mpi_ctx_.local_rank_);
#if NCCL_MINOR < 7
  InitPeerCommunicator();
#endif
  const char* bluefog_num_finalizer_threads =
      std::getenv("BLUEFOG_NUM_FINALIZER_THREADS");
  const int num_finalizer_threads =
      bluefog_num_finalizer_threads == nullptr
          ? 50
          : std::strtol(bluefog_num_finalizer_threads, nullptr, 10);

  finalizer_thread_pool.create(num_finalizer_threads);
  Status timeline_status = GetBluefogTimeline(timeline_ptr_);
  if (!timeline_status.ok()) {
    BFLOG(INFO) << "Timeline is not used because " << timeline_status.reason();
  }
}

#if NCCL_MINOR < 7
void NCCLController::InitPeerCommunicator() {
  int nDevices = 0;
  CUDACHECK(cudaGetDeviceCount(&nDevices));
  CUDACHECK(cudaSetDevice(mpi_ctx_.local_rank_ % nDevices));

  BFLOG(DEBUG) << "Initiate peer communicator";

  // First make pairs that require to build communicator.
  std::vector<std::pair<int, int>> pairs;
  for (int peer_rank : mpi_ctx_.neighbor_out_ranks_) {
    if (mpi_ctx_.rank_ < peer_rank) {
      pairs.push_back(std::make_pair(mpi_ctx_.rank_, peer_rank));
    } else if (mpi_ctx_.rank_ > peer_rank) {
      pairs.push_back(std::make_pair(peer_rank, mpi_ctx_.rank_));
    }
  }
  for (int peer_rank : mpi_ctx_.neighbor_in_ranks_) {
    if (mpi_ctx_.rank_ < peer_rank) {
      pairs.push_back(std::make_pair(mpi_ctx_.rank_, peer_rank));
    } else if (mpi_ctx_.rank_ > peer_rank) {
      pairs.push_back(std::make_pair(peer_rank, mpi_ctx_.rank_));
    }
  }

  // Note our graph definition is directional while the comm is bi-directional.
  // So we make all pairs and sort them by <smaller rank, larger rank>, then
  // deduplicate them.
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  if (mpi_ctx_.rank_ == 2)
    BFLOG(DEBUG, mpi_ctx_.rank_) << "pairs: " << pairs.size();
  for (const auto& pair : pairs) {
    if (mpi_ctx_.rank_ == 2)
      BFLOG(DEBUG, mpi_ctx_.rank_)
          << "pairs: (" << pair.first << "." << pair.second << ")";
    int my_pair_rank = pair.first == mpi_ctx_.rank_ ? 0 : 1;
    int tag = pair.first + pair.second * mpi_ctx_.size_;
    ncclUniqueId nccl_id;
    if (my_pair_rank == 0) {
      ncclGetUniqueId(&nccl_id);
      MPICHECK(MPI_Send((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, pair.second,
                        tag, MPI_COMM_WORLD));
    } else {
      MPICHECK(MPI_Recv((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, pair.first,
                        tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    ncclComm_t new_pair_nccl_comm;
    cudaStream_t new_pair_stream;
    NCCLCHECK(ncclCommInitRank(&new_pair_nccl_comm, 2, nccl_id, my_pair_rank));
    CUDACHECK(cudaStreamCreate(&new_pair_stream));
    nccl_ctx_.nccl_pair_comms[pair] = new_pair_nccl_comm;
    nccl_ctx_.pair_streams[pair] = new_pair_stream;
    nccl_ctx_.pair_order.push_back(pair);
  }
  if (mpi_ctx_.rank_ == 2) BFLOG(DEBUG, mpi_ctx_.rank_) << "pairs end!";
  nccl_ctx_.is_peer_initialized = true;
}

void NCCLController::DestroyPeerCommunicator() {
  BFLOG(DEBUG) << "Destroy peer communicator";
  nccl_ctx_.CleanPeerCommunicator();
}
#endif

bool CheckSameRecvSize(const int* recvcounts, const int size) {
  int first_recv_count;
  for (int i = 0; i < size; i++) {
    if (i == 0) first_recv_count = recvcounts[0];
    if (recvcounts[i] != first_recv_count) return false;
  }
  return true;
}

void NCCLController::Allgather(TensorTableEntry& entry) {
  // Unlike MPI allgatherv, which supports the allgather with different sizes,
  // NCCL require all to be the same size. We just use same routine to find the
  // recvcounts and displacement but displacement is not used.
  int* recvcounts = new int[mpi_ctx_.size_];
  int* displcmnts = new int[mpi_ctx_.size_];
  mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GLOBAL);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GLOBAL);
  if (!CheckSameRecvSize(recvcounts, mpi_ctx_.size_)) {
    throw std::runtime_error(
        "ncclAllGather doesn't support varying lenght of vector. Please make "
        "sure the size of tensors is the same among all processes.");
  }

  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();
  void* buffer_data = (void*)entry.output->data();

  timeline_ptr_->ActivityStart(entry.tensor_name, "COMM. (NCCL)");

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  ncclResult_t ret_code = ncclAllGather(sendbuf, buffer_data, num_elements,
                                        GetNCCLDataType(entry.output),
                                        nccl_ctx_.nccl_comm, nccl_ctx_.stream);
  if (ret_code != ncclSuccess) {
    throw std::runtime_error(
        "ncclAllGather failed, see NCCL output (NCCL_DEBUG=INFO) for details.");
  }

  auto tid = std::this_thread::get_id();
  finalizer_thread_pool.execute([this, entry, tid]() mutable {
    with_device device_guard(entry.device);
    cudaEvent_t event;
    CUDACHECK(this->nccl_ctx_.GetCudaEvent(&event));
    CUDACHECK(cudaEventRecord(event, this->nccl_ctx_.stream));
    CUDACHECK(cudaEventSynchronize(event));
    this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);

    CUDACHECK(this->nccl_ctx_.ReleaseCudaEvent(event));
    entry.callback(Status::OK());
  });

  delete[] recvcounts;
  delete[] displcmnts;
}

void NCCLController::Allreduce(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  void* buffer_data = (void*)entry.output->data();
  int num_elements = entry.tensor->shape().num_elements();

  with_device device_guard(entry.device);

  timeline_ptr_->ActivityStart(entry.tensor_name, "COMM. (NCCL)");
  ncclResult_t ret_code = ncclAllReduce(sendbuf, buffer_data, num_elements,
                                        GetNCCLDataType(entry.tensor), ncclSum,
                                        nccl_ctx_.nccl_comm, nccl_ctx_.stream);
  if (ret_code != ncclSuccess) {
    throw std::runtime_error(
        "ncclAllReduce failed, see NCCL output (NCCL_DEBUG=INFO) for details.");
  }

  auto tid = std::this_thread::get_id();
  finalizer_thread_pool.execute([this, entry, tid]() mutable {
    with_device device_guard(entry.device);
    cudaEvent_t event;
    CUDACHECK(this->nccl_ctx_.GetCudaEvent(&event));
    CUDACHECK(cudaEventRecord(event, this->nccl_ctx_.stream));
    CUDACHECK(cudaEventSynchronize(event));
    this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);

    CUDACHECK(this->nccl_ctx_.ReleaseCudaEvent(event));
    entry.callback(Status::OK());
  });
}

void NCCLController::Broadcast(TensorTableEntry& entry) {
  const int root_rank = entry.root_rank;
  // On root rank, ncclBcast sends data, on other ranks it receives data.
  int num_elements = entry.tensor->shape().num_elements();
  void* data_ptr;
  if (mpi_ctx_.rank_ == root_rank) {
    data_ptr = (void*)entry.tensor->data();
  } else {
    data_ptr = (void*)entry.output->data();
  }

  with_device device_guard(entry.device);

  timeline_ptr_->ActivityStart(entry.tensor_name, "COMM. (NCCL)");
  ncclResult_t ret_code =
      ncclBcast(data_ptr, num_elements, GetNCCLDataType(entry.tensor),
                root_rank, nccl_ctx_.nccl_comm, nccl_ctx_.stream);
  if (ret_code != ncclSuccess) {
    throw std::runtime_error(
        "ncclBroadcast failed, see NCCL output (NCCL_DEBUG=INFO) for details.");
  }

  auto tid = std::this_thread::get_id();
  finalizer_thread_pool.execute([this, entry, tid]() mutable {
    with_device device_guard(entry.device);
    cudaEvent_t event;
    CUDACHECK(this->nccl_ctx_.GetCudaEvent(&event));
    CUDACHECK(cudaEventRecord(event, this->nccl_ctx_.stream));
    CUDACHECK(cudaEventSynchronize(event));
    this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);

    CUDACHECK(this->nccl_ctx_.ReleaseCudaEvent(event));
    entry.callback(Status::OK());
  });
}

void NCCLController::NeighborAllgather(TensorTableEntry& entry) {
  // Note the order of recvcounts and displcments is by the oder of
  // mpi_ctx_.neighbor_in_ranks_ because of MPI_Neighbor_allgather order is
  // determined by then the order of the  values in sources and destinations is
  // identical to the input that was used by the process with the same rank in
  // comm_old in the creation call if the communicator was created with
  // MPI_Dist_graph_create_adjacent.
  int* recvcounts = new int[mpi_ctx_.neighbor_indgree_];
  int* displcmnts = new int[mpi_ctx_.neighbor_indgree_];
  if (!mpi_ctx_.IsTopoSetup()) {
    throw std::runtime_error(
        "Topology has not been set yet cannot run neighbor_allgather");
  }
  mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GRAPH);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GRAPH);
  if (!CheckSameRecvSize(recvcounts, mpi_ctx_.neighbor_indgree_)) {
    throw std::runtime_error(
        "Neighbor_allgather/allreduce doesn't support varying lenght of "
        "vector. Please make "
        "sure the size of tensors is the same among all processes.");
  }

  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();
  void* buffer_data = (void*)entry.output->data();

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  timeline_ptr_->ActivityStart(entry.tensor_name, "COMM. (NCCL)");
  // Pitfall: neighbor_allgather do not include itself.
#if NCCL_MINOR > 6
  ncclGroupStart();
  for (int i = 0; i < mpi_ctx_.neighbor_indgree_; i++) {
    int recv_rank = mpi_ctx_.neighbor_in_ranks_[i];
    int recv_count = recvcounts[i];
    int target_disp = displcmnts[i];
    int element_size = mpi_ctx_.GetMPITypeSize(
        entry.tensor->dtype());  // Assume NCCL use same size as MPI
    void* recvbuf =
        (void*)(static_cast<char*>(buffer_data) + target_disp * element_size);
    NCCLCHECK(ncclRecv(recvbuf, recv_count, GetNCCLDataType(entry.tensor),
                       recv_rank, nccl_ctx_.nccl_comm, nccl_ctx_.stream));
  }
  for (int send_rank : mpi_ctx_.neighbor_out_ranks_) {
    NCCLCHECK(ncclSend(sendbuf, num_elements, GetNCCLDataType(entry.tensor),
                       send_rank, nccl_ctx_.nccl_comm, nccl_ctx_.stream));
  }
  ncclGroupEnd();

  auto tid = std::this_thread::get_id();
  finalizer_thread_pool.execute([this, entry, tid]() mutable {
    with_device device_guard(entry.device);
    cudaEvent_t event;
    CUDACHECK(this->nccl_ctx_.GetCudaEvent(&event));
    CUDACHECK(cudaEventRecord(event, this->nccl_ctx_.stream));
    CUDACHECK(cudaEventSynchronize(event));
    this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);

    CUDACHECK(this->nccl_ctx_.ReleaseCudaEvent(event));
    this->timeline_ptr_->ActivityStart(entry.tensor_name, "CALLBACK", &tid);
    entry.callback(Status::OK());
    this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);
  });

#else
  ncclGroupStart();
  uint recv_rank_index = 0;
  uint send_rank_index = 0;
  for (const auto& pair : nccl_ctx_.pair_order) {
    int peer_rank = mpi_ctx_.rank_ == pair.first ? pair.second : pair.first;

    int send_rank = mpi_ctx_.neighbor_out_ranks_[send_rank_index];
    int recv_rank = mpi_ctx_.neighbor_in_ranks_[recv_rank_index];
    int recv_count = recvcounts[recv_rank_index];
    int target_disp = displcmnts[recv_rank_index];
    bool should_recv = false;
    bool should_send = false;
    if (send_rank_index < mpi_ctx_.neighbor_out_ranks_.size() &&
        peer_rank == send_rank) {
      send_rank_index++;
      should_send = true;
      BFLOG(DEBUG, mpi_ctx_.rank_) << "Should send to rank " << peer_rank;
    }
    if (recv_rank_index < mpi_ctx_.neighbor_in_ranks_.size() &&
        peer_rank == recv_rank) {
      recv_rank_index++;
      should_recv = true;
      BFLOG(DEBUG, mpi_ctx_.rank_) << "Should recv from rank " << peer_rank;
    }

    int element_size = mpi_ctx_.GetMPITypeSize(
        entry.tensor->dtype());  // Assume NCCL use same size as MPI
    void* recvbuf =
        (void*)(static_cast<char*>(buffer_data) + target_disp * element_size);

    if (mpi_ctx_.rank_ == pair.second) {
      // Recv then send
      if (should_recv)
        NCCLCHECK(ncclRecvByBcast(recvbuf, recv_count,
                                  GetNCCLDataType(entry.tensor), recv_rank));

      if (should_send)
        NCCLCHECK(ncclSendByBcast(sendbuf, num_elements,
                                  GetNCCLDataType(entry.tensor), send_rank));
    } else {
      // Send then recv
      if (should_send)
        NCCLCHECK(ncclSendByBcast(sendbuf, num_elements,
                                  GetNCCLDataType(entry.tensor), send_rank));

      if (should_recv)
        NCCLCHECK(ncclRecvByBcast(recvbuf, recv_count,
                                  GetNCCLDataType(entry.tensor), recv_rank));
    }
  }
  ncclGroupEnd();

  for (const auto& stream : nccl_ctx_.pair_streams) {
    CUDACHECK(cudaStreamSynchronize(stream.second));
  }
  entry.callback(Status::OK());
#endif

  delete[] recvcounts;
  delete[] displcmnts;
}

// TODO(ybc) Support partial send and recieve for dyanmic topology usage.
void NCCLController::NeighborAllreduce(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  const int num_elements = entry.tensor->shape().num_elements();
  const int element_size = mpi_ctx_.GetMPITypeSize(
      entry.tensor->dtype());  // Assume NCCL use same size as MPI

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  // MPI have no neighbor_allreduce API. So we will utilize neighbor_allgather.
  // Allgather output will have shape of:
  // (sum of first dimension of every tensor) x (tensor slice shape).
  // For allreduce, the first dimension of every tensor should be the same.
  TensorShape output_shape;
  const int neighbor_size = entry.send_neighbors->empty()
                                ? mpi_ctx_.neighbor_indgree_
                                : entry.recv_neighbors->size();
  const int total_entry_dimension_size =
      entry.tensor->shape().dim_size(0) * neighbor_size;
  output_shape.AddDim(total_entry_dimension_size);
  for (int i = 1; i < entry.tensor->shape().dims(); ++i) {
    output_shape.AddDim(entry.tensor->shape().dim_size(i));
  }

  timeline_ptr->ActivityStart(entry.tensor_name, "ALLOCATE_OUTPUT");
  Status status = entry.context->AllocateOutput(output_shape, &entry.output);
  timeline_ptr->ActivityEnd(entry.tensor_name);

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  // If only partial sending is enabled, the following code block checks whether
  // the sending and recieving neighbors match each other when enable_topo_check
  // is set to be True.
  bool is_topo_check_fail = CheckNeighborSendRecvPattern(
      mpi_ctx_.size_, entry, timeline_ptr,
      mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL));
  if (is_topo_check_fail) {
    entry.callback(Status::InvalidArgument(
        "Send and recv neighbors dont' match in neighbor "
        "allreduce with partial send/recv request."));
  }

  timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
#if NCCL_MINOR > 6
  ncclGroupStart();
  if (entry.send_neighbors->empty()) {
    for (int i = 0; i < mpi_ctx_.neighbor_indgree_; i++) {
      int recv_rank = mpi_ctx_.neighbor_in_ranks_[i];
      void* recvbuf = (void*)(static_cast<const char*>(entry.output->data()) +
                              num_elements * i * element_size);
      NCCLCHECK(ncclRecv(recvbuf, num_elements, GetNCCLDataType(entry.tensor),
                         recv_rank, nccl_ctx_.nccl_comm, nccl_ctx_.stream));
    }
    for (int send_rank : mpi_ctx_.neighbor_out_ranks_) {
      NCCLCHECK(ncclSend(sendbuf, num_elements, GetNCCLDataType(entry.tensor),
                         send_rank, nccl_ctx_.nccl_comm, nccl_ctx_.stream));
    }
  } else {
    for (size_t i = 0; i < entry.recv_neighbors->size(); ++i) {
      int recv_rank = entry.recv_neighbors->at(i);
      void* recvbuf = (void*)(static_cast<const char*>(entry.output->data()) +
                              num_elements * i * element_size);
      NCCLCHECK(ncclRecv(recvbuf, num_elements, GetNCCLDataType(entry.tensor),
                         recv_rank, nccl_ctx_.nccl_comm, nccl_ctx_.stream));
    }
    for (int send_rank : *entry.send_neighbors) {
      NCCLCHECK(ncclSend(sendbuf, num_elements, GetNCCLDataType(entry.tensor),
                         send_rank, nccl_ctx_.nccl_comm, nccl_ctx_.stream));
    }
  }
  ncclGroupEnd();

  auto tid = std::this_thread::get_id();
  finalizer_thread_pool.execute([this, entry, tid]() mutable {
    with_device device_guard(entry.device);
    cudaEvent_t event;
    CUDACHECK(this->nccl_ctx_.GetCudaEvent(&event));
    CUDACHECK(cudaEventRecord(event, this->nccl_ctx_.stream));
    CUDACHECK(cudaEventSynchronize(event));
    this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);

    CUDACHECK(this->nccl_ctx_.ReleaseCudaEvent(event));
    this->timeline_ptr_->ActivityStart(entry.tensor_name, "CALLBACK", &tid);
    entry.callback(Status::OK());
    this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);
  });
#else
  ncclGroupStart();
  uint recv_rank_index = 0;
  uint send_rank_index = 0;
  int send_rank, recv_rank;
  int num_recv_size, num_send_size;
  if (entry.send_neighbors->empty()) {
    num_recv_size = mpi_ctx_.neighbor_in_ranks_.size();
    num_send_size = mpi_ctx_.neighbor_out_ranks_.size();
  } else {
    num_recv_size = entry.recv_neighbors->size();
    num_send_size = entry.send_neighbors->size();
  }
  for (const auto& pair : nccl_ctx_.pair_order) {
    int peer_rank = mpi_ctx_.rank_ == pair.first ? pair.second : pair.first;
    if (entry.send_neighbors->empty()) {
      send_rank = mpi_ctx_.neighbor_out_ranks_[send_rank_index];
      recv_rank = mpi_ctx_.neighbor_in_ranks_[recv_rank_index];
    } else {
      send_rank = entry.send_neighbors->at(send_rank_index);
      recv_rank = entry.recv_neighbors->at(recv_rank_index);
    }

    int target_disp = displcmnts[recv_rank_index];
    bool should_recv = false;
    bool should_send = false;
    if (send_rank_index < num_send_size && peer_rank == send_rank) {
      send_rank_index++;
      should_send = true;
      BFLOG(DEBUG, mpi_ctx_.rank_) << "Should send to rank " << peer_rank;
    }
    if (recv_rank_index < num_recv_size && peer_rank == recv_rank) {
      recv_rank_index++;
      should_recv = true;
      BFLOG(DEBUG, mpi_ctx_.rank_) << "Should recv from rank " << peer_rank;
    }

    void* recvbuf = (void*)(static_cast<const char*>(buffer_data) +
                              num_elements * recv_rank_index * element_size);
    if (mpi_ctx_.rank_ == pair.second) {
      // Recv then send
      if (should_recv)
        NCCLCHECK(ncclRecvByBcast(recvbuf, num_elements,
                                  GetNCCLDataType(entry.tensor), recv_rank));

      if (should_send)
        NCCLCHECK(ncclSendByBcast(sendbuf, num_elements,
                                  GetNCCLDataType(entry.tensor), send_rank));
    } else {
      // Send then recv
      if (should_send)
        NCCLCHECK(ncclSendByBcast(sendbuf, num_elements,
                                  GetNCCLDataType(entry.tensor), send_rank));

      if (should_recv)
        NCCLCHECK(ncclRecvByBcast(recvbuf, num_elements,
                                  GetNCCLDataType(entry.tensor), recv_rank));
    }
  }
  ncclGroupEnd();

  for (const auto& stream : nccl_ctx_.pair_streams) {
    CUDACHECK(cudaStreamSynchronize(stream.second));
  }
  entry.callback(Status::OK());
#endif
}

ncclResult_t NCCLController::ncclSendByBcast(const void* sendbuf,
                                             const int count,
                                             ncclDataType_t data_type,
                                             int peer_rank) {
  int root_rank = -1;
  std::pair<int, int> pair;
  if (mpi_ctx_.rank_ < peer_rank) {
    root_rank = 0;
    pair = std::make_pair(mpi_ctx_.rank_, peer_rank);
  } else {
    root_rank = 1;
    pair = std::make_pair(peer_rank, mpi_ctx_.rank_);
  }
  auto comm_it = nccl_ctx_.nccl_pair_comms.find(pair);
  if (comm_it == nccl_ctx_.nccl_pair_comms.end()) {
    std::string pair_str = "(" + std::to_string(pair.first) + "," +
                           std::to_string(pair.second) + ")";
    throw std::runtime_error(
        pair_str + "cannot be found in the nccl pair communicator when send");
  }
  auto stream_it = nccl_ctx_.pair_streams.find(pair);

  ncclResult_t res = ncclBcast((void*)sendbuf, count, data_type, root_rank,
                               comm_it->second, stream_it->second);
  return res;
}

ncclResult_t NCCLController::ncclRecvByBcast(void* recvbuf, const int count,
                                             ncclDataType_t data_type,
                                             int peer_rank) {
  int root_rank = -1;
  std::pair<int, int> pair;
  if (mpi_ctx_.rank_ < peer_rank) {
    root_rank = 1;
    pair = std::make_pair(mpi_ctx_.rank_, peer_rank);
  } else {
    root_rank = 0;
    pair = std::make_pair(peer_rank, mpi_ctx_.rank_);
  }
  auto comm_it = nccl_ctx_.nccl_pair_comms.find(pair);
  if (comm_it == nccl_ctx_.nccl_pair_comms.end()) {
    std::string pair_str = "(" + std::to_string(pair.first) + "," +
                           std::to_string(pair.second) + ")";
    throw std::runtime_error(
        "cannot be found in the nccl pair communicator when recv");
  }
  auto stream_it = nccl_ctx_.pair_streams.find(pair);

  ncclResult_t res = ncclBcast(recvbuf, count, data_type, root_rank,
                               comm_it->second, stream_it->second);
  return res;
}

}  // namespace common
}  // namespace bluefog
