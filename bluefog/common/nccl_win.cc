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

namespace bluefog {
namespace common {

std::string NCCLWinRequest::to_string() {
  return "Request vector length " + std::to_string(length) + " window_id " +
         std::to_string(name_id);
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
        "vector is not 5");
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
  int id = it->second;
  name_to_id_.erase(it);
  id_to_name_.erase(id_to_name_.find(id));

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
  return id_to_name_.at(id);
}

int NCCLWindowIdManager::GetIdByName(const std::string& name) {
  return name_to_id_.at(name);
}

NCCLWindowManager::~NCCLWindowManager() { FreeWindow(); }

bool NCCLWindowManager::InitializeWinMemory(
    std::shared_ptr<Tensor> tensor,
    std::vector<std::shared_ptr<Tensor>> neighbor_tensors, const int device) {
  self_wins_tensor_ = tensor;
  wins_tensor_vec_ = neighbor_tensors;
  device_ = device;
}

void NCCLWindowManager::FreeWindow() {
  wins_tensor_vec_.clear();
  mutex_mem_.reset();
}

bool NCCLWindowManager::InitializeMutexWin() {
  // We only need one value for self mutex.
  mutex_mem_ =
      std::unique_ptr<int>(new int(0));  // make_unique is c++14 feature.
  return true;
}

bool NCCLWindowManager::DestroyMutexWin() {
  mutex_mem_.reset();
  return true;
}

}  // namespace common
}  // namespace bluefog