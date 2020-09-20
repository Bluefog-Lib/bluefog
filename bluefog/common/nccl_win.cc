// Copyright (C) 2020 Bluefog Team. All Rights Reserved.
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

#include "mpi.h"

#include "logging.h"
#include "nccl_win.h"
#include "operations.h"

namespace bluefog {
namespace common {

std::string NCCLWinRequest::to_string() {
  return "Request vector length " + std::to_string(length) + " window_id " +
         std::to_string(name_id) + " " + DataType_Name(data_type) +
         " OpType: " + std::to_string(to_underlying(op_type));
}

std::vector<int> SerializeNCCLWinRequest(const NCCLWinRequest& req) {
  std::vector<int> res;
  res.push_back(req.length);
  res.push_back(req.name_id);
  res.push_back(to_underlying(req.data_type));
  res.push_back(to_underlying(req.op_type));
  return res;
}

NCCLWinRequest DeserializeNCCLWinRequest(const std::vector<int>& vec) {
  if (vec.size() != 4) {
    throw std::runtime_error(
        "Try to deserialize NCCL win request. But the length of receiving "
        "vector is not 4");
  }
  NCCLWinRequest req;
  req.length = vec[0];
  req.name_id = vec[1];
  req.data_type = static_cast<DataType>(vec[2]);
  req.op_type = static_cast<MPIOpsType>(vec[3]);
  return req;
}

int NCCLWindowIdManager::AllocateId() {
  int id = last_id_.fetch_add(1) + 1;
  return id;
}

Status NCCLWindowIdManager::RegisterIdAndName(int id, const std::string& name) {
  std::lock_guard<std::mutex> guard(mutex_);
  name_to_id_[name] = id;
  id_to_name_[id] = name;
  return Status::OK();
}

Status NCCLWindowIdManager::UnregisterName(const std::string& name) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = name_to_id_.find(name);
  if (it == name_to_id_.end()) {
    return Status::InvalidArgument("Cannot find name " + name +
                                   " in Window Id Manager");
  }
  name_to_id_.erase(it);

  int id = it->second;
  auto id_to_name_it = id_to_name_.find(id);
  if (id_to_name_it == id_to_name_.end()) {
    return Status::InvalidArgument("Cannot find id " + std::to_string(id) +
                                   " to name in Window Id Manager");
  }
  id_to_name_.erase(id_to_name_it);

  return Status::OK();
}

Status NCCLWindowIdManager::CheckNameRegistered(const std::string& name) {
  auto it = name_to_id_.find(name);
  if (it == name_to_id_.end()) {
    return Status::InvalidArgument("Cannot find name " + name +
                                   " in Window Id Manager");
  }
  return Status::OK();
}

Status NCCLWindowIdManager::CheckIdRegistered(int id) {
  auto it = id_to_name_.find(id);
  if (it == id_to_name_.end()) {
    return Status::InvalidArgument("Cannot find id " + std::to_string(id) +
                                   " in Window Id Manager");
  }
  return Status::OK();
}

std::string NCCLWindowIdManager::GetNameById(int id) {
  auto it = id_to_name_.find(id);
  if (it == id_to_name_.end()) {
    throw std::runtime_error("Cannot get window name for id " +
                             std::to_string(id) +
                             " , which should never happen.");
  }
  return id_to_name_.at(id);
}

int NCCLWindowIdManager::GetIdByName(const std::string& name) {
  auto it = name_to_id_.find(name);
  if (it == name_to_id_.end()) {
    throw std::runtime_error("Cannot get window id for name " + name +
                             " , which should never happen.");
  }
  return name_to_id_.at(name);
}

NCCLWindowManager::~NCCLWindowManager() { FreeWindow(); }

bool NCCLWindowManager::InitializeWinMemory(
    std::shared_ptr<Tensor> tensor,
    std::vector<std::shared_ptr<Tensor>> neighbor_tensors, const int device,
    const MPIContext& mpi_ctx) {
  self_wins_tensor_ = tensor;
  device_ = device;
  int neighbor_tensor_index = 0;
  for (int rank = 0; rank < mpi_ctx.size_; rank++) {
    if (rank == mpi_ctx.rank_) {  // self
      wins_tensor_vec_.push_back(tensor);
    } else if (std::find(mpi_ctx.neighbor_in_ranks_.begin(),
                         mpi_ctx.neighbor_in_ranks_.end(),
                         rank) != mpi_ctx.neighbor_in_ranks_.end()) {
      // Neighbor
      wins_tensor_vec_.push_back(neighbor_tensors[neighbor_tensor_index++]);
    } else {
      // Just put as placeholder.
      wins_tensor_vec_.push_back(nullptr);
    }
  }
}

void NCCLWindowManager::FreeWindow() {
  wins_tensor_vec_.clear();
}

// TODO(ybc) Following code is duplicated from MPI version.
bool NCCLWindowManager::InitializeMutexWin() {
  int self_rank = 0;
  int global_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &self_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_size);
  if (global_size <= 1) {
    // We don't need any mutex for this case.
    return false;
  }

  if (!mutex_win_) {
     mutex_win_  = std::make_shared<MPI_Win>();
  }
  // We only need one value for self mutex.
  mutex_mem_.resize(global_size);
  std::fill_n(mutex_mem_.data(), global_size, 0);

  int element_size = 0;
  MPI_Type_size(MPI_INT, &element_size);
  int win_size = global_size * element_size;
  MPI_Win_create((void *)mutex_mem_.data(), win_size, element_size, MPI_INFO_NULL, MPI_COMM_WORLD,
                 mutex_win_.get());
  return true;
}

bool NCCLWindowManager::DestroyMutexWin() {
  if (!mutex_win_) {
    mutex_mem_.clear();
    return false;
  }
  MPI_Win_free(mutex_win_.get());
  mutex_win_.reset();
  mutex_mem_.clear();
  return true;
}

std::shared_ptr<MPI_Win> NCCLWindowManager::GetMutexWin() { return mutex_win_; }

}  // namespace common
}  // namespace bluefog