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

#include "nccl_win.h"

namespace bluefog {
namespace common {

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