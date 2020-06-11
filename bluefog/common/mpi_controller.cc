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
#include <thread>

#include "cuda_util.h"
#include "operations.h"
#include "timeline.h"

namespace bluefog {
namespace common {

// Due to unclear reason that mpi_put/get/accumlate under the
// mpi_lock epoch cannot send too long vector in one time, we
// define this number as the maximum size of win_ops can send.
static const char* BLUEFOG_MAX_WIN_SENT = std::getenv("BLUEFOG_MAX_WIN_SENT_LENGTH");
static const int MAX_WIN_SENT =
    BLUEFOG_MAX_WIN_SENT == nullptr
        ? 2000
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

  BFLOG(DEBUG) << "MPI controller initialized.";
}

int MPIController::GetTypeSize(DataType dtype) {
  return mpi_ctx_.GetMPITypeSize(dtype);
}

void MPIController::Allgather(TensorTableEntry& entry) {
  int* recvcounts = new int[mpi_ctx_.size_];
  int* displcmnts = new int[mpi_ctx_.size_];
  mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GLOBAL);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GLOBAL);

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

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);
  int ret_code = MPI_Allreduce(
      sendbuf, buffer_data, num_elements, mpi_ctx_.GetMPIDataType(entry.tensor),
      MPI_SUM, mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL));
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
  mpi_ctx_.DestroyWinMutex();
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

  mpi_ctx_.InitializeWinMutex();
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
  mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GRAPH);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GRAPH);

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

void MPIController::NeighborAllreduce(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

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

  
  timeline_ptr->ActivityStart(entry.tensor_name, "ALLOCATE_OUTPUT");
  Status status = entry.context->AllocateOutput(output_shape, &entry.output);
  timeline_ptr->ActivityEnd(entry.tensor_name);

  void* buffer_data = (void*)entry.output->data();

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");
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
  timeline_ptr->ActivityEnd(entry.tensor_name);

  timeline_ptr->ActivityStart(entry.tensor_name, "COMPUTE_AVERAGE");
  entry.callback(Status::OK());
  timeline_ptr->ActivityEnd(entry.tensor_name);
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

Status MPIController::WinCreate(
    std::shared_ptr<Tensor> tensor,
    std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
    const std::string& name, const int device) {

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  timeline_ptr->ActivityStart(name, "WIN_CREATE");
  // We need to explicitly set the device here.
  with_device device_guard(device);
  // 1. Regist a Name and create a window first.
  if (!mpi_ctx_.RegisterWindowName(name)) {
    return Status::InvalidArgument(std::string("Win_create failed with ") +
                                   name);
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
  // For example: size=4 power two ring topology
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

  return Status::OK();
}

Status MPIController::WinFree(const std::string& name, int device) {
  if (!mpi_ctx_.UnregisterWindowName(name)) {
    return Status::InvalidArgument(std::string("Win_free failed with ") + name);
  }
  return Status::OK();
}

Status MPIController::WinFreeAll() {
  if (!mpi_ctx_.UnregisterAllWindowName()) {
    return Status::InvalidArgument(std::string("Win_free_all failed."));
  }
  BFLOG(DEBUG) << "All MPI Win has been freed.";
  return Status::OK();
}

Status MPIController::WinSync(const std::string& name, int device) {
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
    throw std::runtime_error(std::string("Cannot find ") + entry.tensor_name);
  }
  std::shared_ptr<WindowManager> win_mananger = it->second;
  MPI_Win mpi_win = *(win_mananger->GetWinByRank(mpi_ctx_.rank_));

  std::vector<int> mutex_ranks = {};  // used in mutex only.

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  std::vector<std::pair<int, double>> sorted_dst_weights =
      GetSortedDstWeights(mpi_ctx_.rank_, mpi_ctx_.size_, entry.dst_weights);

  for (auto kv : sorted_dst_weights) {
    int target_rank = kv.first;
    double weight = kv.second;

    BFLOG(TRACE, mpi_ctx_.rank_) << "Start MPI_Put for " << entry.tensor_name << " to " << target_rank;

    if (entry.require_mutex) {
      mutex_ranks.clear();
      mutex_ranks.push_back(target_rank);
      timeline_ptr->ActivityStart(entry.tensor_name, "Aquire_Mutex");
      WinMutexAcquire(mutex_ranks, /*is_sync=*/false);
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

    if (entry.require_mutex) {
      WinMutexRelease(mutex_ranks, /*is_sync=*/false);
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
    throw std::runtime_error(std::string("Cannot find ") + entry.tensor_name);
  }
  std::shared_ptr<WindowManager> win_mananger = it->second;
  MPI_Win mpi_win = *(win_mananger->GetWinByRank(mpi_ctx_.rank_));

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);
  std::vector<int> mutex_ranks = {};  // used in mutex only.

  std::vector<std::pair<int, double>> sorted_dst_weights =
      GetSortedDstWeights(mpi_ctx_.rank_, mpi_ctx_.size_, entry.dst_weights);

  for (auto kv : sorted_dst_weights) {
    int target_rank = kv.first;
    double weight = kv.second;
    // avoid putting the tensor for itself (NOT valid).
    if (target_rank == mpi_ctx_.rank_) continue;

    if (entry.require_mutex) {
      mutex_ranks.clear();
      mutex_ranks.push_back(target_rank);
      timeline_ptr->ActivityStart(entry.tensor_name, "Aquire_Mutex");
      WinMutexAcquire(mutex_ranks, /*is_sync=*/false);
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
        if (entry.require_mutex) WinMutexRelease(mutex_ranks, /*is_sync=*/false);
        throw std::runtime_error(
            "MPI_Accumulate failed, see MPI output for details.");
      }
      target_disp += sent_size;
      sent_size = std::min(MAX_WIN_SENT, num_elements - target_disp);
    }
    MPI_Win_unlock(target_rank, mpi_win);

    timeline_ptr->ActivityEnd(entry.tensor_name);
    if (entry.require_mutex) {
      WinMutexRelease(mutex_ranks, /*is_sync=*/false);
    }
  }
  BFLOG(TRACE, mpi_ctx_.rank_) << "MPI_Accmulate for " << entry.tensor_name
                      << " is done.";

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
                             std::string(" in registered win object name."));
  }
  std::shared_ptr<WindowManager> win_mananger = it->second;
  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);
  std::vector<int> mutex_ranks = {};  // used in mutex only.

  MPI_Win mpi_win = *(win_mananger->GetGlobalWin());
  for (auto kv : entry.src_weights) {
    int target_rank = kv.first;
    // avoid getting the tensor for itself.
    if (target_rank == mpi_ctx_.rank_) continue;

    if (entry.require_mutex) {
      mutex_ranks.clear();
      mutex_ranks.push_back(target_rank);
      timeline_ptr->ActivityStart(entry.tensor_name, "Aquire_Mutex");
      WinMutexAcquire(mutex_ranks, /*is_sync=*/false);
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

    if (entry.require_mutex) {
      WinMutexRelease(mutex_ranks, /*is_sync=*/false);
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

Status MPIController::WinMutexAcquire(const std::vector<int>& acquire_ranks,
                                      bool is_sync) {
  // The logic is similar to read-write lock:
  // Value starts at 0:
  //    1. is_sync step +1 if value is 0 else wait
  //    2. Not is_sync step -1 if value is <= 0 else wait. (i.e. we allow
  //    multiple non sync step).
  if (mpi_ctx_.win_mutex.empty()) {
    return Status::PreconditionError(
        "Cannot accquire Win Mutex because it may not be initialized or has "
        "been destroyed.");
  }

  // TODO(ybc) Try better implementation than Spin Lock.
  // Recall that we build N windows across all N processes.
  // The spin value is stored in the rank i for i-th window.
  // Other process will got to acquire it.
  int one = 1;
  int minus_one = -1;
  int oldval = 0;
  int target_disp = 0;

  std::shared_ptr<MPI_Win> mutex_win;

  for (int rank : acquire_ranks) {
    mutex_win = mpi_ctx_.win_mutex[rank];
    BFLOG(TRACE, mpi_ctx_.rank_) << "Acquire Win Mutex for rank " << rank;
    MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, *mutex_win);
    do {
      if (is_sync) {
        MPI_Fetch_and_op(&one, &oldval, MPI_INT, rank, target_disp, MPI_SUM,
                        *mutex_win);
        MPI_Win_flush(rank, *mutex_win);
        if (oldval == 0) break;
        MPI_Accumulate(&minus_one, 1, MPI_INT, rank, target_disp, 1, MPI_INT,
                      MPI_SUM, *mutex_win);
        MPI_Win_flush(rank, *mutex_win);
      } else {
        MPI_Fetch_and_op(&minus_one, &oldval, MPI_INT, rank, target_disp, MPI_SUM,
                        *mutex_win);
        MPI_Win_flush(rank, *mutex_win);
        if (oldval <= 0) break;
        MPI_Accumulate(&one, 1, MPI_INT, rank, target_disp, 1, MPI_INT,
                      MPI_SUM, *mutex_win);
        MPI_Win_flush(rank, *mutex_win);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    } while (1);
    MPI_Win_unlock(rank, *mutex_win);
  }

  return Status::OK();
}

Status MPIController::WinMutexRelease(const std::vector<int>& release_ranks,
                                      bool is_sync) {
  int one = 1;
  int minus_one = -1;
  int target_disp = 0;

  if (mpi_ctx_.win_mutex.empty()) {
    return Status::PreconditionError(
        "Cannot release Win Mutex because it may not be initialized or has "
        "been destroyed.");
  }

  std::shared_ptr<MPI_Win> mutex_win;
  for (int rank : release_ranks) {
    mutex_win = mpi_ctx_.win_mutex[rank];
    BFLOG(TRACE, mpi_ctx_.rank_) << "Release Win Mutex for rank " << rank;
    MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, *mutex_win);
    if (is_sync) {
      // TODO(ybc) Notice the following accumulate may cause the value to be
      // negative, i.e. more release ops is called than acquire.
      MPI_Accumulate(&minus_one, 1, MPI_INT, rank, target_disp, 1, MPI_INT,
                     MPI_SUM, *mutex_win);
    } else {
      MPI_Accumulate(&one, 1, MPI_INT, rank, target_disp, 1, MPI_INT, MPI_SUM,
                     *mutex_win);
    }
    MPI_Win_unlock(rank, *mutex_win);
  }

  return Status::OK();
}

WinMutexGuard::WinMutexGuard(MPIController* mpi_controller,
                             const std::vector<int>& acquire_ranks,
                             bool is_sync)
    : mpi_controller_(mpi_controller),
      acquire_ranks_(acquire_ranks),
      is_sync_(is_sync) {
  mpi_controller_->WinMutexAcquire(acquire_ranks_, is_sync_);
}

WinMutexGuard::~WinMutexGuard() {
  mpi_controller_->WinMutexAcquire(acquire_ranks_, is_sync_);
}

}  // namespace common
}  // namespace bluefog
