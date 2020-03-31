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

#ifndef BLUEFOG_COMMON_GLOBAL_STATE_H
#define BLUEFOG_COMMON_GLOBAL_STATE_H

#include <atomic>
#include <queue>
#include <thread>

#include "tensor_queue.h"
#include "mpi_controller.h"

#include "timeline.h"

namespace bluefog {
namespace common {

// The global state shared by threads.
//
// MPI is a library that stores a lot of global per-program state and often
// requires running on a single thread. As a result, we have to have a single
// background thread responsible for all MPI operations, and communicate with
// that background thread through global state.
struct BluefogGlobalState {
  // An atomic boolean which is set to true when background thread is started.
  // This ensures that only one background thread is spawned.
  std::atomic_flag initialize_flag = ATOMIC_FLAG_INIT;

  // Background thread running MPI communication.
  std::thread background_thread;

  // Whether the background thread should shutdown.
  std::atomic_bool shut_down{false};

  // Whether collective context has been completed on the background thread.
  std::atomic_bool initialization_done{false};

  // Timeline writer.
  Timeline timeline;

  // Flag indicating whether timeline enabled.
  bool timeline_enabled = false;

  std::shared_ptr<MPIController> controller;

  TensorQueue tensor_queue;

  ~BluefogGlobalState() {
    // Make sure that the destructor of the background thread is safe to
    // call. If a thread is still joinable (not detached or complete) its
    // destructor cannot be called.
    if (background_thread.joinable()) {
      shut_down = true;
      background_thread.join();
    }
  }
};

}  // namespace common
}  // namespace bluefog

#endif