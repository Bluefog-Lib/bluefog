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

static const char* BF_MAX_SLEEP_USEC_FOR_WIN_PASSIVE = std::getenv("BLUEFOG_SLEEP_USEC_FOR_WIN_PASSIVE");
static const int MAX_SLEEP_USEC_FOR_WIN_PASSIVE =
    BF_MAX_SLEEP_USEC_FOR_WIN_PASSIVE == nullptr
        ? 1000
        : std::strtol(BF_MAX_SLEEP_USEC_FOR_WIN_PASSIVE, nullptr, 10);

constexpr int BFWinPassiveSuccess = 0;
constexpr int BFWinPassiveFail = 1;
constexpr int BFWinPassiveRetry = 2;

const int kWinPassiveRecvRequestTag = 20201234;
const int kWinPassiveRecvAckTag = 20200827;
const int kWinPassiveDoneTag = 20200913;

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  return GetNCCLDataType(tensor->dtype());
}

ncclDataType_t GetNCCLDataType(const DataType bf_data_type) {
  switch (bf_data_type) {
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
      throw std::logic_error("Type " + DataType_Name(bf_data_type) +
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
  self_rank = rank;
  self_local_rank = local_rank;

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

  cuda_device = local_rank % nDevices;
  is_initialized = true;
  return;
}

#if NCCL_MINOR < 7
void NCCLContext::CleanPeerCommunicators() {
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
  is_peer_comm_initialized = false;
}
#endif

void NCCLContext::CleanWindowCommunicators() {
  for (int i = 0; i < (int)nccl_win_active_comms.size(); i++) {
    CUDACHECK(cudaStreamDestroy(nccl_win_active_streams[i]));
    CUDACHECK(cudaStreamDestroy(nccl_win_passive_streams[i]));
    if (i != self_rank) {
      NCCLCHECK(ncclCommDestroy(nccl_win_active_comms[i]));
      NCCLCHECK(ncclCommDestroy(nccl_win_passive_comms[i]));
    }
  }
  nccl_win_active_streams.clear();
  nccl_win_passive_streams.clear();
  nccl_win_active_comms.clear();
  nccl_win_passive_comms.clear();
  is_window_comm_initialized = false;
}

void NCCLContext::Finalize() {
  if (win_passive_recv_initialized) {
    win_passive_recv_shutdown.store(true);
    while(!win_passive_recv_shutdown_done) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }
  finalizer_thread_pool.reset();

  NCCLCHECK(ncclCommDestroy(nccl_comm));
  CUDACHECK(cudaStreamDestroy(stream));

#if NCCL_MINOR < 7
  CleanPeerCommunicator();
#endif
  CleanWindowCommunicators();

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

  nccl_ctx_.finalizer_thread_pool.create(num_finalizer_threads);
  Status timeline_status = GetBluefogTimeline(timeline_ptr_);
  if (!timeline_status.ok()) {
    BFLOG(INFO) << "Timeline is not used because " << timeline_status.reason();
  }
  // Barrier helps NCCL to synchronize after initialization and avoid
  // deadlock that we've been seeing without it.
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
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
  for (const auto& pair : pairs) {
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
  nccl_ctx_.is_peer_comm_initialized = true;
}

void NCCLController::DestroyPeerCommunicators() {
  BFLOG(DEBUG) << "Destroy peer communicators";
  nccl_ctx_.CleanPeerCommunicators();
}
#endif

void NCCLController::InitWindowCommunicators() {
  if (nccl_ctx_.is_window_comm_initialized) {
    BFLOG(DEBUG)
        << "NCCL context for window communicator has been initialized but"
        << "InitWindowCommunicators is called again";
    return;
  }
  BFLOG(DEBUG) << "Initiate window communicators";
  // Assume one device per process
  int nDevices = 0;
  CUDACHECK(cudaGetDeviceCount(&nDevices));
  CUDACHECK(cudaSetDevice(mpi_ctx_.local_rank_ % nDevices));

  int greatest_priority;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));

  for (int i = 0; i < mpi_ctx_.size_; i++) {
    cudaStream_t new_window_active_stream;
    cudaStream_t new_window_passive_stream;
    CUDACHECK(cudaStreamCreateWithPriority(
        &new_window_active_stream, cudaStreamNonBlocking, greatest_priority));
    CUDACHECK(cudaStreamCreateWithPriority(
        &new_window_passive_stream, cudaStreamNonBlocking, greatest_priority));
    nccl_ctx_.nccl_win_active_streams.push_back(new_window_active_stream);
    nccl_ctx_.nccl_win_passive_streams.push_back(new_window_passive_stream);
  }
  assert(nccl_ctx_.nccl_win_active_streams.size() == mpi_ctx_.size_);
  assert(nccl_ctx_.nccl_win_passive_streams.size() == mpi_ctx_.size_);

  // Each active/passive communicator is connected with two nodes only.
  // New rank for self is always 0 in acitive comm and 1 in passive comm.
  // First loop will buid  (X for null, A for active, and P for passive)
  //    0  1  2
  // 0  X  P  P
  // 1  A  X  P
  // 2  A  A  X
  // Each A and P is connected in off-diagonal direction.
  // Second loop build in a reverse way
  //    0  1  2
  // 0  X  A  A
  // 1  P  X  A
  // 2  P  P  X
  // When it finished, rank 1 will have two vectors looks like [A X A] and [P X P].
  BFLOG(DEBUG) << "Initiate pair window communicator for ncclReduce usage.";
  nccl_ctx_.nccl_win_active_comms.resize(mpi_ctx_.size_);
  nccl_ctx_.nccl_win_passive_comms.resize(mpi_ctx_.size_);
  for (int i = 0; i < mpi_ctx_.size_; i++) {
    ncclUniqueId nccl_id;
    ncclComm_t nccl_win_accum_comm;
    if (mpi_ctx_.rank_ == i) {
      // Self to self, so just an empty one.
    } else if (mpi_ctx_.rank_ > i) {
      int tag = i + mpi_ctx_.rank_ * mpi_ctx_.size_;
      ncclGetUniqueId(&nccl_id);
      MPICHECK(MPI_Send((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, i,
                        tag, MPI_COMM_WORLD));
      NCCLCHECK(ncclCommInitRank(&nccl_win_accum_comm, 2, nccl_id, /*rank=*/1));
      nccl_ctx_.nccl_win_passive_comms[i] = nccl_win_accum_comm;
    } else {
      int tag = mpi_ctx_.rank_ + i * mpi_ctx_.size_;
      MPICHECK(MPI_Recv((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, i,
                        tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      NCCLCHECK(ncclCommInitRank(&nccl_win_accum_comm, 2, nccl_id, /*rank=*/0));
      nccl_ctx_.nccl_win_active_comms[i] = nccl_win_accum_comm;
    }
  }
  // Second time, make it in a reverse way.
  for (int i = 0; i < mpi_ctx_.size_; i++) {
    ncclUniqueId nccl_id;
    ncclComm_t nccl_win_accum_comm;
    if (mpi_ctx_.rank_ == i) {
      // Self to self, so just an empty one.
    } else if (mpi_ctx_.rank_ > i) {
      // Whoever has bigger rank number, the new rank is 1 in accum_comm.
      int tag = i + mpi_ctx_.rank_ * mpi_ctx_.size_;
      ncclGetUniqueId(&nccl_id);
      MPICHECK(MPI_Send((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, i,
                        tag, MPI_COMM_WORLD));
      NCCLCHECK(ncclCommInitRank(&nccl_win_accum_comm, 2, nccl_id, /*rank=*/0));
      nccl_ctx_.nccl_win_active_comms[i] = nccl_win_accum_comm;
    } else {
      int tag = mpi_ctx_.rank_ + i * mpi_ctx_.size_;
      MPICHECK(MPI_Recv((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, i,
                        tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      NCCLCHECK(ncclCommInitRank(&nccl_win_accum_comm, 2, nccl_id, /*rank=*/1));
      nccl_ctx_.nccl_win_passive_comms[i] = nccl_win_accum_comm;
    }
  }
}

void NCCLController::DestroyWindowCommunicators() {
  BFLOG(DEBUG) << "Destroy window communicators";
  nccl_ctx_.CleanWindowCommunicators();
}

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
  Status status = mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GLOBAL);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GLOBAL);
  if (!CheckSameRecvSize(recvcounts, mpi_ctx_.size_)) {
    delete[] recvcounts;
    delete[] displcmnts;
    entry.callback(Status::PreconditionError(
        "ncclAllGather doesn't support varying lenght of vector. Please make "
        "sure the size of tensors is the same among all processes."));
    return;
  }
  if (!status.ok()) {
    delete[] recvcounts;
    delete[] displcmnts;
    entry.callback(status);
    return;
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
  nccl_ctx_.finalizer_thread_pool.execute([this, entry, tid]() mutable {
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
    std::string error_msg =
        "ncclAllReduce failed, see NCCL output (NCCL_DEBUG=INFO) "
        "for details.";
    BFLOG(ERROR) << error_msg;
    entry.callback(Status::UnknownError(error_msg));
    return;
  }

  auto tid = std::this_thread::get_id();
  nccl_ctx_.finalizer_thread_pool.execute([this, entry, tid]() mutable {
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
    std::string error_msg =
        "ncclBcast failed, see NCCL output (NCCL_DEBUG=INFO) "
        "for details.";
    BFLOG(ERROR) << error_msg;
    entry.callback(Status::UnknownError(error_msg));
    return;
  }

  auto tid = std::this_thread::get_id();
  nccl_ctx_.finalizer_thread_pool.execute([this, entry, tid]() mutable {
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
  Status status = mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GRAPH);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GRAPH);
  if (!CheckSameRecvSize(recvcounts, mpi_ctx_.neighbor_indgree_)) {
    delete[] recvcounts;
    delete[] displcmnts;
    entry.callback(Status::PreconditionError(
        "Neighbor_allgather/allreduce doesn't support varying lenght of "
        "vector. Please make "
        "sure the size of tensors is the same among all processes."));
    return;
  }
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
  nccl_ctx_.finalizer_thread_pool.execute([this, entry, tid]() mutable {
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

void NCCLController::NeighborAllreduce(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  const int num_elements = entry.tensor->shape().num_elements();
  const int element_size = mpi_ctx_.GetMPITypeSize(
      entry.tensor->dtype());  // Assume NCCL use same size as MPI

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  // NCCL does not have neighbor_allreduce API. So neighbor_allgather 
  // is implemented through Send/Recv first.
  // Allgather output will have shape of:
  // (sum of first dimension of every tensor) x (tensor slice shape).
  // For allreduce, the first dimension of every tensor should be the same.
  // Assume the memory has already been allocated at python side.

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
    return; 
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
    int rank = mpi_ctx_.rank_;
    int size = mpi_ctx_.size_;
    std::sort(entry.send_neighbors->begin(), entry.send_neighbors->end(),
              [rank, size](int a, int b) {
                int a_index = a >= rank ? a - rank : a - rank + size;
                int b_index = b >= rank ? b - rank : b - rank + size;
                return a_index - b_index;
              });
    std::sort(entry.recv_neighbors->begin(), entry.recv_neighbors->end(),
              [rank, size](int a, int b) {
                int a_index = a >= rank ? a - rank : a - rank + size;
                int b_index = b >= rank ? b - rank : b - rank + size;
                return b_index - a_index;
              });
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
  nccl_ctx_.finalizer_thread_pool.execute([this, entry, tid]() mutable {
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

#if NCCL_MINOR < 7
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
#endif

void WinPassiveRecvRequest(int self_rank, NCCLContext& nccl_ctx) {
  BFLOG(TRACE, self_rank) << "WinPassiveRecvRequest thread initialized";
  std::vector<int> req_buf(4, -1);
  std::vector<int> res_buf(1, -1);
  cudaStream_t win_stream;
  while (!nccl_ctx.win_passive_recv_shutdown) {
    MPI_Status mpi_status;
    MPI_Request mpi_request;
    int mpi_flag;
    // receive message from any source
    MPI_Irecv(req_buf.data(), 4, MPI_INT, MPI_ANY_SOURCE,
              kWinPassiveRecvRequestTag, MPI_COMM_WORLD, &mpi_request);
    while (!nccl_ctx.win_passive_recv_shutdown) {
      MPI_Test(&mpi_request, &mpi_flag, &mpi_status);
      if (!mpi_flag) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      } else {
        break; // Successfully received somthing.
      }
    }
    if (nccl_ctx.win_passive_recv_shutdown) {
      MPI_Cancel(&mpi_request);
      BFLOG(TRACE, self_rank) << "WinPassiveRecvRequest is shutting down.";
      break;
    }
    if(mpi_status.MPI_ERROR != MPI_SUCCESS) {
      std::string error_message;
      error_message += "MPI_Irecv from " + std::to_string(mpi_status.MPI_SOURCE);
      error_message += "; with error code " + std::to_string(mpi_status.MPI_ERROR);
      BFLOG(ERROR) << "Error encountered in WinPassiveRecvRequest: "
                   << error_message;
    }

    // Received and succeed.
    int source = mpi_status.MPI_SOURCE;
    NCCLWinRequest req = DeserializeNCCLWinRequest(req_buf);
    bool self_processing = source == self_rank;
    if (self_processing) {
      BFLOG(ERROR) << "WinPassiveRecvRequest recieved request to process self memeory.";
    }

    Status status = nccl_ctx.window_id_manager.CheckIdRegistered(req.name_id);
    if ((mpi_status.MPI_ERROR != MPI_SUCCESS) || !status.ok() || self_processing) {
      res_buf[0] = BFWinPassiveFail;
      MPICHECK(MPI_Send(res_buf.data(), 1, MPI_INT, source,
                        kWinPassiveRecvAckTag, MPI_COMM_WORLD));
      if (status.ok()) {
        BFLOG(DEBUG) << "WinPassiveRecvRequest request refused due be previous show error";
      } else {
        BFLOG(DEBUG) << "WinPassiveRecvRequest request refused because " << status.reason();
      }
      continue;
    }
    if (nccl_ctx.nccl_win_mutex.try_lock()) {
      res_buf[0] = BFWinPassiveSuccess;
      MPICHECK(MPI_Send(res_buf.data(), 1, MPI_INT, source,
                        kWinPassiveRecvAckTag, MPI_COMM_WORLD));
      BFLOG(TRACE, self_rank) << "Recv and be able to process request from "
                              << source << ": " << req.to_string();
    } else {
      res_buf[0] = BFWinPassiveRetry;
      MPICHECK(MPI_Send(res_buf.data(), 1, MPI_INT, source,
                        kWinPassiveRecvAckTag, MPI_COMM_WORLD));
      BFLOG(TRACE, self_rank) << "Due to lock, unable to process request from "
                              << source << ": " << req.to_string();
      continue;
    }
  
    std::string win_name = nccl_ctx.window_id_manager.GetNameById(req.name_id);
    std::shared_ptr<NCCLWindowManager> nccl_win_manager =
        nccl_ctx.named_win_map.at(win_name);
    with_device device_guard(nccl_win_manager->GetWinMemoryDevice());
    if (req.op_type == MPIOpsType::WIN_PUT) {
      void* recvbuf = (void*)nccl_win_manager->GetWinMemoryByRank(source);
      auto& win_comm = nccl_ctx.nccl_win_passive_comms[source];
      win_stream = nccl_ctx.nccl_win_passive_streams[source];

#if NCCL_MINOR > 6
      // Self_rank for passive is always 1 so that source is always 0.
      NCCLCHECK(ncclRecv(recvbuf, req.length, GetNCCLDataType(req.data_type),
                         /*source=*/0, win_comm, win_stream));
#else
      NCCLCHECK(ncclBroadcast(/*sendbuf=*/nullptr, /*recvbuf=*/recvbuf,
                              num_elements, GetNCCLDataType(entry.tensor),
                              /*root=*/0, win_comm, win_stream));
#endif
    } else if (req.op_type == MPIOpsType::WIN_GET) {
      void* sendbuf = (void*)nccl_win_manager->GetWinMemoryByRank(self_rank);
      auto& win_comm = nccl_ctx.nccl_win_passive_comms[source];
      win_stream = nccl_ctx.nccl_win_passive_streams[self_rank];
#if NCCL_MINOR > 6
      // Self_rank for passive is always 1 so that destination is always 0.
      NCCLCHECK(ncclSend(sendbuf, req.length, GetNCCLDataType(req.data_type),
                         /*dest=*/0, win_comm, win_stream));
#else
      NCCLCHECK(ncclBroadcast(/*sendbuf=*/sendbuf, /*recvbuf=*/nullptr,
                              num_elements, GetNCCLDataType(entry.tensor),
                              /*root=*/1, win_comm, win_stream));
#endif
    } else if (req.op_type == MPIOpsType::WIN_ACCUMULATE) {
      void* recvbuf = (void*)nccl_win_manager->GetWinMemoryByRank(source);
      auto& win_comm = nccl_ctx.nccl_win_passive_comms[source]; // Self_rank for passive is always 1
      win_stream = nccl_ctx.nccl_win_passive_streams[source]; 
      NCCLCHECK(ncclReduce(/*sendbuf=*/recvbuf, /*recv=*/recvbuf, req.length,
                           GetNCCLDataType(req.data_type), ncclSum, 
                           /*root=*/1, win_comm, win_stream));
    } else {
      BFLOG(ERROR) << "Receive wrong ops types in WinPassiveRecvRequest: "
                   << to_underlying(req.op_type)
                   << ". Supporting types are WIN_PUT = 6,WIN_GET = 7, and "
                      "WIN_ACCUMULATE = 8,";
    }
    nccl_ctx.nccl_win_mutex.unlock();
    nccl_ctx.finalizer_thread_pool.execute([win_stream, req, source]() mutable {
      CUDACHECK(cudaStreamSynchronize(win_stream));
      if (req.op_type == MPIOpsType::WIN_ACCUMULATE ||
          req.op_type == MPIOpsType::WIN_PUT) {
        int done = 1;
        MPICHECK(MPI_Send(&done, 1, MPI_INT, source, kWinPassiveDoneTag,
                          MPI_COMM_WORLD));
      }
    });
  }
  nccl_ctx.finalizer_thread_pool.reset(); // Need to wait until all finalizer_thread_pool stopped.
  nccl_ctx.win_passive_recv_shutdown_done = true;
}

void NCCLController::WinCreate(TensorTableEntry& entry) {
  const std::string& name = entry.tensor_name;
  if (!nccl_ctx_.win_passive_recv_initialized) {
    nccl_ctx_.win_passive_recv_thread =
        std::thread(WinPassiveRecvRequest, mpi_ctx_.rank_, std::ref(nccl_ctx_));
    nccl_ctx_.win_passive_recv_initialized = true;
    nccl_ctx_.win_passive_recv_thread.detach();
  }
  if (!nccl_ctx_.is_window_comm_initialized) {
    InitWindowCommunicators();
    nccl_ctx_.is_window_comm_initialized = true;
  }

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  timeline_ptr->ActivityStart(name, "WIN_CREATE");
  // We need to explicitly set the device here.
  with_device device_guard(entry.device);
  // 1. Check the name is used or not.
  auto it = nccl_ctx_.named_win_map.find(name);
  if (it != nccl_ctx_.named_win_map.end()) {
    entry.callback(
        Status::InvalidArgument(std::string("Win_create failed with ") + name));
    return;
  }

  // 2. Create a NCCL Window Manager.
  auto nccl_window = std::make_shared<NCCLWindowManager>();
  nccl_window->InitializeWinMemory(entry.tensor, entry.neighbor_tensors, entry.device, mpi_ctx_);
  nccl_window->InitializeMutexWin();

  // 3. Registered NCCL window manager and allocate unique id for them.
  nccl_ctx_.named_win_map[name] = nccl_window;

  int window_id;
  if (mpi_ctx_.rank_ == 0) {
    window_id = nccl_ctx_.window_id_manager.AllocateId();
  }
  MPICHECK(MPI_Bcast((void*)&window_id, 1, MPI_INT, 0, MPI_COMM_WORLD));
  nccl_ctx_.window_id_manager.RegisterIdAndName(window_id, name);
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  timeline_ptr->ActivityEnd(name);

  entry.callback(Status::OK());
}

// Helper function to Execute WinFree.
Status NCCLController::WinFreeReturnStatus(TensorTableEntry& entry) {
  const std::string& name = entry.tensor_name;
  auto it = nccl_ctx_.named_win_map.find(name);
  if (it == nccl_ctx_.named_win_map.end()) {
    return Status::InvalidArgument(std::string("Win_free failed with ") + name);
  }
  int window_id;
  if (mpi_ctx_.rank_ == 0) {
    window_id = nccl_ctx_.window_id_manager.GetIdByName(name);
  }
  MPICHECK(MPI_Bcast((void*)&window_id, 1, MPI_INT, 0, MPI_COMM_WORLD));
  int my_window_id = nccl_ctx_.window_id_manager.GetIdByName(name);
  if (my_window_id != window_id) {
    return Status::InvalidArgument(
        "Different processes tried to free different window name " + name);
  }
  it->second->DestroyMutexWin();
  it->second->FreeWindow();
  nccl_ctx_.named_win_map.erase(it);
  if (!nccl_ctx_.window_id_manager.UnregisterName(name).ok()) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    return Status::InvalidArgument(std::string("Win_free failed with ") + name);
  } else {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
  return Status::OK();
}

void NCCLController::WinFree(TensorTableEntry& entry) {
  Status status = WinFreeReturnStatus(entry);
  entry.callback(status);
}

void NCCLController::WinFreeAll(TensorTableEntry& entry) {
  std::vector<std::string> win_names;
  win_names.reserve(nccl_ctx_.named_win_map.size());
  for (auto& it : nccl_ctx_.named_win_map) {
    win_names.push_back(it.first);
  }
  TensorTableEntry entry_for_win_free = entry;
  std::sort(win_names.begin(), win_names.end());
  Status status;
  for (auto& name : win_names) {
    entry_for_win_free.tensor_name = name;
    status = WinFreeReturnStatus(entry_for_win_free);
    if (!status.ok()) {
      entry.callback(status);
      return;
    }
  }
  BFLOG(DEBUG) << "All NCCL Wins have been freed.";
  entry.callback(Status::OK());
}

Status NCCLController::WinSync(const std::string& name, int device, bool with_associated_p) {
  auto it = nccl_ctx_.named_win_map.find(name);
  if (it == nccl_ctx_.named_win_map.end()) {
    return Status::InvalidArgument(std::string("Win_sync failed with ") + name);
  }
  if(with_associated_p) {
    BFLOG(WARNING) << "With Associated P is not supported in NCCL implementation yet.";
  }
  // We don't need to do anything since our mimic window don't do anything on
  // seperate memory.
  return Status::OK();
}

void NCCLController::WinPut(TensorTableEntry& entry) {
  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  int num_elements = entry.tensor->shape().num_elements();
  DataType data_type = entry.tensor->dtype();
  auto it = nccl_ctx_.named_win_map.find(entry.tensor_name);
  if (it == nccl_ctx_.named_win_map.end()) {
    throw std::runtime_error(std::string("Cannot find ") + entry.tensor_name +
                             " in (NCCL) registered win name.");
  }
  int window_name_id =
      nccl_ctx_.window_id_manager.GetIdByName(entry.tensor_name);

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  // Because the weighted tensor is a tempory memory that won't be hold by entry
  // and the callback will be finish in another thread. Hence, if we didn't make
  // it an copy of shared ptr into callback. thess tensors will be destroyed.
  std::vector<std::shared_ptr<Tensor>> weight_tensor_holder;

  std::unique_lock<std::mutex> lock_win_passive(nccl_ctx_.nccl_win_mutex);
  ncclGroupStart();
  // TODO(ybc) sort the dst_weights?
  for (auto kv : entry.dst_weights) {
    int target_rank = kv.first;
    double weight = kv.second;
    // 1. Talk with passive recv thread to open the request first.
    NCCLWinRequest nccl_win_request = {.length = num_elements,
                                       .name_id = window_name_id,
                                       .data_type = data_type,
                                       .op_type = MPIOpsType::WIN_PUT};
    const std::vector<int> req_buf = SerializeNCCLWinRequest(nccl_win_request);
    std::vector<int> res_buf(1, -1);

    // To avoid dead lock, we will use mutex to exclude the order of active thread
    // and passive thread on nccl communication ops. An exclusive communication can
    // be established only when self active thread get lock and peer passive thread
    // get the lock as well.
    // Note the live lock may NOT be able to avoid in thic approach.
    do {
      BFLOG(TRACE, mpi_ctx_.rank_)
          << "Send request to " << target_rank << ": "
          << DeserializeNCCLWinRequest(req_buf).to_string();
      MPICHECK(MPI_Send(req_buf.data(), 4, MPI_INT, target_rank,
                        kWinPassiveRecvRequestTag, MPI_COMM_WORLD));
      MPICHECK(MPI_Recv(res_buf.data(), 1, MPI_INT, target_rank,
                        kWinPassiveRecvAckTag, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
      if (res_buf[0] == BFWinPassiveFail) {
        // TODO(ybc) How to handle it??
        throw std::runtime_error(
            "Failed after the passive recv thread for NCCL Win_put.");
      }
      if (res_buf[0] == BFWinPassiveRetry) {
        lock_win_passive.unlock();
        unsigned int random_usec =
            (rand() + (mpi_ctx_.rank_ * 123)) % MAX_SLEEP_USEC_FOR_WIN_PASSIVE;
        std::this_thread::sleep_for(std::chrono::microseconds(random_usec));
        lock_win_passive.lock();
      }
    } while (res_buf[0] != BFWinPassiveSuccess);

    if (entry.require_mutex) {
      timeline_ptr->ActivityStart(entry.tensor_name, "Aquire_Mutex");
      WinMutexAcquire(entry.tensor_name, {target_rank}, /*is_sync=*/false);
      timeline_ptr->ActivityEnd(entry.tensor_name);
    }
    // 2. Passive recv thread is ready to recv
    // TODO(ybc) implement mutex for nccl window
    timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
    // avoid putting the tensor for itself (NOT valid).
    if (target_rank == mpi_ctx_.rank_) continue;
    std::shared_ptr<Tensor> tensor = entry.tensor->data_weight(weight);
    weight_tensor_holder.push_back(tensor);
    void* sendbuf = (void*)tensor->data();
    auto& win_comm = nccl_ctx_.nccl_win_active_comms[target_rank];
    auto& win_stream = nccl_ctx_.nccl_win_active_streams[mpi_ctx_.rank_];
#if NCCL_MINOR > 6
    // Self rank in active comms is always 0 and pair rank is 1.
    NCCLCHECK(ncclSend(sendbuf, num_elements, GetNCCLDataType(entry.tensor),
                       /*dest=*/1, win_comm, win_stream));
#else
    NCCLCHECK(ncclBroadcast(sendbuf, /*recvbuf=*/nullptr, num_elements,
                            GetNCCLDataType(entry.tensor),
                            /*root=*/0, win_comm, win_stream));
#endif
  }
  ncclGroupEnd();
  lock_win_passive.unlock();

  // 3. Confirm the recv side is done as well. (Otherwise the mutex may be problematic)
  std::vector<MPI_Request> requests(entry.dst_weights.size());
  std::vector<MPI_Status> statuses(entry.dst_weights.size());
  std::vector<int> done(entry.dst_weights.size(), -1);
  int count = 0;
  for (auto kv : entry.dst_weights) {
    int target_rank = kv.first;
    MPICHECK(MPI_Irecv(done.data() + count, 1, MPI_INT, target_rank,
                       kWinPassiveDoneTag, MPI_COMM_WORLD, &requests[count]));
    count++;
  }

  auto tid = std::this_thread::get_id();
  cudaEvent_t event;
  auto& win_stream = this->nccl_ctx_.nccl_win_active_streams[this->mpi_ctx_.rank_];
  CUDACHECK(this->nccl_ctx_.GetCudaEvent(&event));
  CUDACHECK(cudaEventRecord(event, win_stream));
  CUDACHECK(cudaEventSynchronize(event));
  MPICHECK(
      MPI_Waitall(entry.dst_weights.size(), requests.data(), statuses.data()));
  this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);  // COMMUNICATE

  if (entry.require_mutex) {
    std::vector<int> dst_ranks;
    for(auto& kv : entry.dst_weights){
      dst_ranks.push_back(kv.first);
    }
    WinMutexRelease(entry.tensor_name, dst_ranks, /*is_sync=*/false);
  }

  CUDACHECK(this->nccl_ctx_.ReleaseCudaEvent(event));
  this->timeline_ptr_->ActivityStart(entry.tensor_name, "CALLBACK", &tid);
  BFLOG(TRACE, this->mpi_ctx_.rank_)
      << "Win_Put(NCCL) for " << entry.tensor_name << " is done.";
  entry.callback(Status::OK());
  this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);
}

void NCCLController::WinAccumulate(TensorTableEntry& entry) {
  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  int num_elements = entry.tensor->shape().num_elements();
  DataType data_type = entry.tensor->dtype();
  auto it = nccl_ctx_.named_win_map.find(entry.tensor_name);
  if (it == nccl_ctx_.named_win_map.end()) {
    throw std::runtime_error(std::string("Cannot find ") + entry.tensor_name +
                             " in (NCCL) registered win name.");
  }
  int window_name_id =
      nccl_ctx_.window_id_manager.GetIdByName(entry.tensor_name);

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  // Because the weighted tensor is a tempory memory that won't be hold by entry
  // and the callback will be finish in another thread. Hence, if we didn't make
  // it an copy of shared ptr into callback. thess tensors will be destroyed.
  std::vector<std::shared_ptr<Tensor>> weight_tensor_holder;
  std::unique_lock<std::mutex> lock_win_passive(nccl_ctx_.nccl_win_mutex);

  ncclGroupStart();
  // TODO(ybc) sort the dst_weights?
  // Make it in parallel for loop??
  for (auto kv : entry.dst_weights) {
    int target_rank = kv.first;
    double weight = kv.second;
    // 1. Talk with passive recv thread to open the request first.
    NCCLWinRequest nccl_win_request = {.length = num_elements,
                                       .name_id = window_name_id,
                                       .data_type = data_type,
                                       .op_type = MPIOpsType::WIN_ACCUMULATE};
    const std::vector<int> req_buf = SerializeNCCLWinRequest(nccl_win_request);
    std::vector<int> res_buf(1, -1);
    // To avoid dead lock, we will use mutex to exclude the order of active thread
    // and passive thread on nccl communication ops. An exclusive communication can
    // be established only when self active thread get lock and peer passive thread
    // get the lock as well.
    // Note the live lock may NOT be able to avoid in thic approach.
    do {
      MPICHECK(MPI_Send(req_buf.data(), 4, MPI_INT, target_rank,
                        kWinPassiveRecvRequestTag, MPI_COMM_WORLD));
      MPICHECK(MPI_Recv(res_buf.data(), 1, MPI_INT, target_rank,
                        kWinPassiveRecvAckTag, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
      if (res_buf[0] == BFWinPassiveFail) {
        // TODO(ybc) How to handle it??
        throw std::runtime_error(
            "Failed after the passive recv thread for NCCL Win_put.");
      }
      if (res_buf[0] == BFWinPassiveRetry) {
        lock_win_passive.unlock();
        unsigned int random_usec =
            (rand() + (mpi_ctx_.rank_ * 123)) % MAX_SLEEP_USEC_FOR_WIN_PASSIVE;
        BFLOG(TRACE, mpi_ctx_.rank_)
            << "Will retry request to " << target_rank << " in "
            << std::to_string(random_usec)
            << "usec: " << DeserializeNCCLWinRequest(req_buf).to_string();
        std::this_thread::sleep_for(std::chrono::microseconds(random_usec));
        lock_win_passive.lock();
      } else {
        BFLOG(TRACE, mpi_ctx_.rank_)
            << "Send and accept request to " << target_rank << ": "
            << DeserializeNCCLWinRequest(req_buf).to_string();
      }
    } while (res_buf[0] != BFWinPassiveSuccess);

    if (entry.require_mutex) {
      timeline_ptr->ActivityStart(entry.tensor_name, "Aquire_Mutex");
      WinMutexAcquire(entry.tensor_name, {target_rank}, /*is_sync=*/false);
      timeline_ptr->ActivityEnd(entry.tensor_name);
    }
    // 2. Passive recv thread is ready to recv
    // TODO(ybc) implement mutex for nccl window
    timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
    // avoid putting the tensor for itself (NOT valid).
    if (target_rank == mpi_ctx_.rank_) continue;
    std::shared_ptr<Tensor> tensor = entry.tensor->data_weight(weight);
    weight_tensor_holder.push_back(tensor);
    void* sendbuf = (void*)tensor->data();
    auto& win_comm = nccl_ctx_.nccl_win_active_comms[target_rank];
    auto& win_stream = nccl_ctx_.nccl_win_active_streams[mpi_ctx_.rank_];

    // Self rank in active comms is always 0 and pair rank is 1.
    // We use ncclReduce to mimic Accumulate, hence, root is 1 (pair rank).
    NCCLCHECK(ncclReduce(sendbuf, /*recv=*/nullptr, num_elements,
                         GetNCCLDataType(entry.tensor), ncclSum,
                         /*root=*/1, win_comm, win_stream));
  }
  ncclGroupEnd();
  lock_win_passive.unlock();

  // 3. Confirm the recv side is done as well. (Otherwise the mutex may be problematic)
  std::vector<MPI_Request> requests(entry.dst_weights.size());
  std::vector<MPI_Status> statuses(entry.dst_weights.size());
  std::vector<int> done(entry.dst_weights.size(), -1);
  int count = 0;
  for (auto kv : entry.dst_weights) {
    int target_rank = kv.first;
    MPICHECK(MPI_Irecv(done.data() + count, 1, MPI_INT, target_rank,
                       kWinPassiveDoneTag, MPI_COMM_WORLD, &requests[count]));
    count++;
  }

  auto tid = std::this_thread::get_id();
  cudaEvent_t event;
  auto& win_stream = this->nccl_ctx_.nccl_win_active_streams[this->mpi_ctx_.rank_];
  CUDACHECK(this->nccl_ctx_.GetCudaEvent(&event));
  CUDACHECK(cudaEventRecord(event, win_stream));
  CUDACHECK(cudaEventSynchronize(event));
  MPICHECK(
      MPI_Waitall(entry.dst_weights.size(), requests.data(), statuses.data()));
  this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);  // COMMUNICATE

  if (entry.require_mutex) {
    std::vector<int> dst_ranks;
    for(auto& kv : entry.dst_weights){
      dst_ranks.push_back(kv.first);
    }
    WinMutexRelease(entry.tensor_name, dst_ranks, /*is_sync=*/false);
  }

  CUDACHECK(this->nccl_ctx_.ReleaseCudaEvent(event));
  this->timeline_ptr_->ActivityStart(entry.tensor_name, "CALLBACK", &tid);
  BFLOG(TRACE, this->mpi_ctx_.rank_)
      << "Win_Put(NCCL) for " << entry.tensor_name << " is done.";
  entry.callback(Status::OK());
  this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);
}

void NCCLController::WinGet(TensorTableEntry& entry) {
  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  auto it = nccl_ctx_.named_win_map.find(entry.tensor_name);
  if (it == nccl_ctx_.named_win_map.end()) {
    throw std::runtime_error(std::string("Cannot find ") + entry.tensor_name +
                             " in (NCCL) registered win name.");
  }
  int window_name_id =
      nccl_ctx_.window_id_manager.GetIdByName(entry.tensor_name);
  std::shared_ptr<NCCLWindowManager> nccl_win_manager =
      nccl_ctx_.named_win_map.at(entry.tensor_name);

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  std::unique_lock<std::mutex> lock_win_passive(nccl_ctx_.nccl_win_mutex);
  ncclGroupStart();
  // TODO(ybc) sort the src_weights?
  for (auto& kv : entry.src_weights) {
    int target_rank = kv.first;

    with_device device_guard(nccl_win_manager->GetWinMemoryDevice());
    std::shared_ptr<Tensor> tensor =
        nccl_win_manager->GetAssociateTensorByRank(target_rank);
    void* recvbuf = (void*)nccl_win_manager->GetWinMemoryByRank(target_rank);
    int num_elements = tensor->shape().num_elements();
    DataType data_type = tensor->dtype();

    // avoid getting the tensor for itself.
    if (target_rank == mpi_ctx_.rank_) continue;

    // 1. Talk with passive recv thread to open the request first.
    NCCLWinRequest nccl_win_request = {.length = num_elements,
                                       .name_id = window_name_id,
                                       .data_type = data_type,
                                       .op_type = MPIOpsType::WIN_GET};
    const std::vector<int> req_buf = SerializeNCCLWinRequest(nccl_win_request);
    std::vector<int> res_buf(1, -1);

    // To avoid dead lock, we will use mutex to exclude the order of active thread
    // and passive thread on nccl communication ops. An exclusive communication can
    // be established only when self active thread get lock and peer passive thread
    // get the lock as well.
    // Note the live lock may NOT be able to avoid in thic approach.
    do {
      BFLOG(TRACE, mpi_ctx_.rank_)
          << "Send request to " << target_rank << ": "
          << DeserializeNCCLWinRequest(req_buf).to_string();
      MPICHECK(MPI_Send(req_buf.data(), 4, MPI_INT, target_rank,
                        kWinPassiveRecvRequestTag, MPI_COMM_WORLD));
      MPICHECK(MPI_Recv(res_buf.data(), 1, MPI_INT, target_rank,
                        kWinPassiveRecvAckTag, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
      if (res_buf[0] == BFWinPassiveFail) {
        // TODO(ybc) How to handle it??
        throw std::runtime_error(
            "Failed after the passive recv thread for NCCL Win_get.");
      }
      if (res_buf[0] == BFWinPassiveRetry) {
        lock_win_passive.unlock();
        unsigned int random_usec =
            (rand() + (mpi_ctx_.rank_ * 123)) % MAX_SLEEP_USEC_FOR_WIN_PASSIVE;
        std::this_thread::sleep_for(std::chrono::microseconds(random_usec));
        lock_win_passive.lock();
      }
    } while (res_buf[0] != BFWinPassiveSuccess);

    if (entry.require_mutex) {
      timeline_ptr->ActivityStart(entry.tensor_name, "Aquire_Mutex");
      WinMutexAcquire(entry.tensor_name, {target_rank}, /*is_sync=*/false);
      timeline_ptr->ActivityEnd(entry.tensor_name);
    }
    // 2. Passive recv thread is ready to recv
    // TODO(ybc) implement mutex for nccl window
    timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
    auto& win_comm = nccl_ctx_.nccl_win_active_comms[target_rank];
    auto& win_stream = nccl_ctx_.nccl_win_active_streams[target_rank];
#if NCCL_MINOR > 6
    // Self rank in active comms is always 0 and pair rank is 1.
    NCCLCHECK(ncclRecv(recvbuf, num_elements, GetNCCLDataType(tensor),
                       /*dest=*/1, win_comm, win_stream));
#else
    NCCLCHECK(ncclBroadcast(nullptr, /*recvbuf=*/recvbuf, num_elements,
                            GetNCCLDataType(entry.tensor),
                            /*root=*/1, win_comm, win_stream));
#endif
  }
  ncclGroupEnd();
  lock_win_passive.unlock();

  auto tid = std::this_thread::get_id();
  std::vector<cudaEvent_t> events;
  for (auto& kv : entry.src_weights) {
    int target_rank = kv.first;
    cudaEvent_t event;
    CUDACHECK(this->nccl_ctx_.GetCudaEvent(&event));
    events.push_back(event);
    auto& win_stream = this->nccl_ctx_.nccl_win_active_streams[target_rank];
    CUDACHECK(cudaEventRecord(event, win_stream));
  }
  for (auto& event : events) {
    CUDACHECK(cudaEventSynchronize(event));
    CUDACHECK(this->nccl_ctx_.ReleaseCudaEvent(event));
  }
  events.clear();
  this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);  // COMMUNICATE

  if (entry.require_mutex) {
    std::vector<int> src_ranks;
    for(auto& kv : entry.src_weights){
      src_ranks.push_back(kv.first);
    }
    WinMutexRelease(entry.tensor_name, src_ranks, /*is_sync=*/false);
  }

  this->timeline_ptr_->ActivityStart(entry.tensor_name, "CALLBACK", &tid);
  BFLOG(TRACE, this->mpi_ctx_.rank_)
      << "Win_Get(NCCL) for " << entry.tensor_name << " is done.";
  entry.callback(Status::OK());
  this->timeline_ptr_->ActivityEnd(entry.tensor_name, &tid);
}

Status NCCLController::WinMutexAcquire(const std::string& name,
                                       const std::vector<int>& acquire_ranks,
                                       bool is_sync) {
  BFLOG(TRACE, mpi_ctx_.rank_) << "Win Mutex (NCCL) for " << name << " is acquired.";
  auto it = nccl_ctx_.named_win_map.find(name);
  if (it == nccl_ctx_.named_win_map.end()) {
    throw std::runtime_error(std::string("Cannot find ") + name +
                             " in (NCCL) registered win name.");
  }
  std::shared_ptr<MPI_Win> mutex_win = it->second->GetMutexWin();
  return MPIWinMutexAcquireImpl(mutex_win, acquire_ranks, mpi_ctx_.rank_, is_sync);
}

Status NCCLController::WinMutexRelease(const std::string& name,
                                       const std::vector<int>& release_ranks,
                                       bool is_sync) {
  BFLOG(TRACE, mpi_ctx_.rank_) << "Win Mutex (NCCL) for " << name << " is released.";
  auto it = nccl_ctx_.named_win_map.find(name);
  if (it == nccl_ctx_.named_win_map.end()) {
    throw std::runtime_error(std::string("Cannot find ") + name +
                             " in (NCCL) registered win name.");
  }
  std::shared_ptr<MPI_Win> mutex_win = it->second->GetMutexWin();
  return MPIWinMutexReleaseImpl(mutex_win, release_ranks, mpi_ctx_.rank_, is_sync);
}

void NCCLController::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
    void*& buffer_data, size_t& buffer_len) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  FusionBufferManager* buffer_manager;
  GetBluefogFusionBuffer(buffer_manager);
  auto buffer = buffer_manager->GetBuffer(first_entry.device);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryInFusionBuffer(e, buffer_data_at_offset);
    offset += e.tensor->size();
  }

  buffer_len = (size_t)offset;

  // Set the input data to originate from the buffer.
  fused_input_data = buffer_data;
}

void NCCLController::MemcpyOutFusionBuffer(
    const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryOutFusionBuffer(buffer_data_at_offset, e);
    offset += e.output->size();
  }
}

void NCCLController::MemcpyEntryInFusionBuffer(const TensorTableEntry& e,
                                               void* buffer_data_at_offset) {
  CUDA_CHECK(cudaMemcpyAsync(buffer_data_at_offset, e.tensor->data(),
                             (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                             nccl_ctx_.stream));
}

void NCCLController::MemcpyEntryOutFusionBuffer(
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  CUDA_CHECK(cudaMemcpyAsync(e.tensor->data(), buffer_data_at_offset,
                             (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                             nccl_ctx_.stream));
}

}  // namespace common
}  // namespace bluefog
