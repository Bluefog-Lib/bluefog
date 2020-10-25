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
#include <chrono>
#include <memory>
#include <queue>
#include <thread>

#include "tensor_queue.h"
#include "mpi_controller.h"
#include "timeline.h"

#if HAVE_NCCL
#include "nccl_controller.h"
#endif

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

  // Background thread cycle time in milliseconds.  Fractional numbers are permitted.
  double cycle_time_ms = 0.5;

  // Time point when last cycle started.
  std::chrono::steady_clock::time_point last_cycle_start;

  // Time point when coordinator last checked for stalled tensors.
  std::chrono::steady_clock::time_point last_stall_check;

  std::shared_ptr<MPIController> controller;

  #if HAVE_NCCL
  std::unique_ptr<NCCLController> nccl_controller;
  #endif

  TensorQueue tensor_queue;

  // Threshold for Tensor Fusion.  All tensors that occupy memory beyond this
  // threshold will be fused.
  int64_t tensor_fusion_threshold = 8 * 1024 * 1024;
  FusionBufferManager fusion_buffer;

  // Because setting topology happens in the main thread instead of communication
  // thread. Following three variables are to sync between them.
  std::atomic_bool setting_topology{false};
  std::atomic_bool setting_topology_done{false};
  std::atomic_bool ready_to_setting_topology{false};

  // Only exists on the coordinator node (rank zero). Maintains a vector of
  // requests to allreduce every tensor (keyed by tensor name).
  // The associated time_point is recorded when the request is received, which
  // is used in stalled tensors check.
  std::unique_ptr<std::unordered_map<
      std::string,
      std::tuple<std::vector<Request>, std::chrono::steady_clock::time_point>>>
      message_table;

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