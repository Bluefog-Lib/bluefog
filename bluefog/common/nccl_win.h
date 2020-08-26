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

#ifndef BLUEFOG_COMMON_NCCL_WIN_H
#define BLUEFOG_COMMON_NCCL_WIN_H

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "common.h"

namespace bluefog {
namespace common {

struct NCCLWinRequest {
  int source;          // Request rank;
  int length;          // the lenght of send vectors.
  int name_id;         // Used to identify which window to use.
  DataType data_type;  // Such as BLUEFOG_FLOAT32, BLUEFOG_FLOAT64, etc.
  MPIOpsType op_type;  // Such as win_put, win_get, or win_accumulate.
};

std::vector<int> SerializeNCCLWinRequest(const NCCLWinRequest& req);

NCCLWinRequest DeserializeNCCLWinRequest(const std::vector<int>& vec);

class NCCLWindowIdManager {
 public:
  // Only Rank 0 will allocate Id.
  int AllocateId();
  Status RegisterIdAndName(int id, const std::string& name);
  Status UnregisterName(const std::string& name);

 private:
  std::atomic_int last_id_;
  // In order to identify the named win, each unique name is associated with an
  // int;
  std::unordered_map<std::string, int> name_to_id_;
  std::unordered_map<int, std::string> id_to_name_;
  std::mutex mutex_;
};

class NCCLWindowManager {
 public:
  NCCLWindowManager() = default;
  ~NCCLWindowManager();

  bool InitializeWinMemory(
      std::shared_ptr<Tensor> tensor,
      std::vector<std::shared_ptr<Tensor>> neighbor_tensors, const int device);

  inline std::shared_ptr<Tensor> GetAssociateTensorByRank(int rank) {
    return wins_tensor_vec_[rank];
  }

  inline const void* GetWinMemoryByRank(int rank) {
    return wins_tensor_vec_[rank]->data();
  }

  // Manually free the win memory.
  void FreeWindow();

  bool InitializeMutexWin();
  bool DestroyMutexWin();

 private:
  std::vector<std::shared_ptr<Tensor>> wins_tensor_vec_;

  std::shared_ptr<Tensor> self_wins_tensor_;

  int device_;

  std::unique_ptr<int> mutex_mem_;
};

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_NCCL_CONTROLLER_H