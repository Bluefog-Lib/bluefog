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

#include "operations.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "common.h"
#include "global_state.h"
#include "logging.h"

// Bluefog knobs.
#define BLUEFOG_TIMELINE "BLUEFOG_TIMELINE"

namespace bluefog {
namespace common {

namespace {

// All the Bluefog state that must be stored globally per-process.
BluefogGlobalState bluefog_global;

MPIContext mpi_context;

}  // namespace

bool RunLoopOnce(BluefogGlobalState& state);

void BackgroundThreadLoop(BluefogGlobalState& state) {
  auto mpi_ctx_manager = MPIContextManager();
  mpi_context.Initialize(std::vector<int>{}, mpi_ctx_manager);

  // Initialize controller
  state.controller->Initialize();

  // Signal that initialization is completed.
  state.initialization_done = true;
  BFLOG(INFO, bluefog_global.controller->GetRank()) << "Bluefog Initialized";

  // Open the timeline file
  char* bluefog_timeline_loc = std::getenv(BLUEFOG_TIMELINE);

  if (bluefog_timeline_loc != nullptr) {
    std::string timeline_filename = std::string(bluefog_timeline_loc) +
                                    std::to_string(bluefog_rank()) +
                                    std::string(".json");
    state.timeline.Initialize(timeline_filename, bluefog_size());

    state.timeline_enabled = true;
  }

  // Iterate until shutdown.
  while (RunLoopOnce(state))
    ;

  BFLOG(DEBUG, bluefog_global.controller->GetRank())
      << "Shutting down background thread";

  // Signal that shutdown has been requested.
  state.shut_down = true;
  // Notify all outstanding operations that Bluefog has been shut down
  // and finalize tensor queue.
  std::vector<StatusCallback> callbacks;
  bluefog_global.tensor_queue.FinalizeTensorQueue(callbacks);
  for (auto& cb : callbacks) {
    cb(SHUT_DOWN_ERROR);
  }
  mpi_context.Finalize(mpi_ctx_manager);
}

bool RunLoopOnce(BluefogGlobalState& state) {
  try {
    auto entry = state.tensor_queue.PopMessagesFromQueue();
    switch (entry.mpi_ops_type) {
      case MPIOpsType::ALLREDUCE:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name;
        state.timeline.ActivityStart(entry.tensor_name, MPI_ALLREDUCE);
        state.controller->Allreduce(entry);
        state.timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::BROADCAST:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name;
        state.timeline.ActivityStart(entry.tensor_name, MPI_BROADCAST);
        state.controller->Broadcast(entry);
        state.timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::ALLGATHER:
        state.timeline.ActivityStart(entry.tensor_name, MPI_ALLGATHER);
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name;
        state.controller->Allgather(entry);
        state.timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::NEIGHBOR_ALLGATHER:
        state.timeline.ActivityStart(entry.tensor_name, MPI_NEIGHBOR_ALLGATHER);
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name;
        state.controller->NeighborAllgather(entry);
        state.timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::NEIGHBOR_ALLREDUCE:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name;
        state.timeline.ActivityStart(entry.tensor_name, MPI_NEIGHBOR_ALLREDUCE);
        state.controller->NeighborAllreduce(entry);
        state.timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::BARRIER:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing Barrier now ";
        state.controller->Barrier(entry);
        break;
      // TODO(ybc) All above Ops are collective ops. If the order
      // is disarranged, the whole process will hang. This is possible in
      // tensorflow. For example, if two ops are not control dependent to each
      // other, the order of allreduce request by them are undeterminisitc.
      case MPIOpsType::WIN_PUT:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing WIN_PUT on " << entry.tensor_name;
        state.timeline.ActivityStart(entry.tensor_name, MPI_WIN_PUT);
        state.controller->WinPut(entry);
        state.timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::WIN_GET:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing WIN_GET on " << entry.tensor_name;
        state.timeline.ActivityStart(entry.tensor_name, MPI_WIN_GET);
        state.controller->WinGet(entry);
        state.timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::WIN_ACCUMULATE:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing WIN_ACCUMULATE on " << entry.tensor_name;
        state.timeline.ActivityStart(entry.tensor_name, MPI_WIN_ACCUMULATE);
        state.controller->WinAccumulate(entry);
        state.timeline.ActivityEnd(entry.tensor_name);
        break;
      default:
        state.timeline.ActivityEnd(entry.tensor_name);  // End activity for enqueue
        throw std::runtime_error("Unsupported/Unkown MPI Operation Types");
    }
  } catch (std::length_error& e) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  } catch (std::exception& e) {
    BFLOG(ERROR) << e.what();
  }
  return !bluefog_global.shut_down;
}

// Start Bluefog background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeBluefogOnce() {
  mpi_context
      .Enable();  // We always enable mpi since we relied on MPI only now.
  if (!bluefog_global.initialize_flag.test_and_set()) {
    bluefog_global.controller.reset(
        new MPIController(bluefog_global.tensor_queue, mpi_context));
    bluefog_global.initialization_done = false;
    bluefog_global.background_thread =
        std::thread(BackgroundThreadLoop, std::ref(bluefog_global));
  }
  // Wait to ensure that the background thread has finished initializing MPI.
  while (!bluefog_global.initialization_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  BFLOG(DEBUG) << "Background thread init done";
}

Status CheckInitialized() {
  if (!bluefog_global.initialization_done) {
    return NOT_INITIALIZED_ERROR;
  }
  return Status::OK();
}

extern "C" {

void bluefog_init() { InitializeBluefogOnce(); }

void bluefog_shutdown() {
  if (bluefog_global.background_thread.joinable()) {
    bluefog_global.shut_down = true;
    bluefog_global.background_thread.join();
    // Reset the initialization flag to allow restarting with bluefog_init(...)
    bluefog_global.initialize_flag.clear();
    bluefog_global.shut_down = false;
  }
}

int bluefog_rank() {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->GetRank();
}

int bluefog_local_rank() {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->GetLocalRank();
}

int bluefog_size() {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->GetSize();
}

int bluefog_local_size() {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->GetLocalSize();
}

int bluefog_neighbor_size() {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->GetNeighborSize();
}

int bluefog_mpi_threads_supported() {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->IsMpiThreadsSupported() ? 1 : 0;
}

int bluefog_unified_mpi_window_model_supported() {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->IsMpiUnifiedModel() ? 1 : 0;
}

int bluefog_set_topology(int indegree, const int* sources, int outdegree,
                         const int* destinations) {
  if (!bluefog_global.initialization_done) {
    BFLOG(ERROR)
        << "Cannot set the topology because bluefog has not been initialized.";
    return -1;
  }
  if (!bluefog_global.controller->IsWinObjetEmpty()) {
    BFLOG(ERROR)
        << "Cannot set the topology because there are window object uncleared.";
    return -1;
  }
  if (bluefog_global.tensor_queue.size() > 0) {
    BFLOG(ERROR)
        << "Cannot set the topology because there are unfinished MPI ops.";
    return -1;
  }
  return bluefog_global.controller->SetTopology(indegree, sources, outdegree,
                                                destinations);
}

int bluefog_set_topology_with_weights(int indegree, const int* sources,
                                      int outdegree, const int* destinations,
                                      float self_weight, const float* neighbor_weights) {
  int ret = bluefog_set_topology(indegree, sources, outdegree, destinations);
  if (ret != 1) {
    return ret;
  }
  return bluefog_global.controller->SetTopologyWeights(indegree, sources,
                                                       self_weight, neighbor_weights);
}

int bluefog_load_topology(int* indegree, int*& sources, int* outdegree,
                          int*& destinations) {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->LoadTopology(indegree, sources, outdegree,
                                                 destinations);
}

int bluefog_load_topology_weights(
    float& self_weight_,
    const std::unordered_map<int, float>*& neighbor_weights_) {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->LoadTopologyWeights(self_weight_, neighbor_weights_);
}


int bluefog_timeline(const bool start_activity, const char* tensor_name,
                     const char* activity_name) {
  if (!bluefog_global.initialization_done) {
    return -1;
  }

  Timeline* timeline_ptr;
  Status status = GetBluefogTimeline(timeline_ptr);
  if (!status.ok()) {
    BFLOG(ERROR) << "Failed to get timeline: " << status.reason();
    return -1;
  }

  if (start_activity) {
    timeline_ptr->ActivityStart(tensor_name, activity_name);
  } else {
    timeline_ptr->ActivityEnd(tensor_name);
  }
  return 1;
}

}  // extern "C"

Status EnqueueTensorAllreduce(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              const std::string& name, const int device,
                              StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.output = output;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::ALLREDUCE;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  return status;
}

Status EnqueueTensorBroadcast(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              const int root_rank, const std::string& name,
                              const int device, StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::BROADCAST;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  return status;
}

Status EnqueueTensorAllgather(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<OpContext> context,
                              const std::string& name, const int device,
                              StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.context = context;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::ALLGATHER;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  return status;
}

Status EnqueueTensorNeighborAllgather(std::shared_ptr<Tensor> tensor,
                                      std::shared_ptr<OpContext> context,
                                      const std::string& name, const int device,
                                      StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.context = context;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::NEIGHBOR_ALLGATHER;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  return status;
}

Status EnqueueTensorNeighborAllreduce(std::shared_ptr<OpContext> context,
                                      std::shared_ptr<Tensor> tensor,
                                      std::shared_ptr<Tensor> output,
                                      const std::string& name, const int device,
                                      StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.output = output;
  e.context = context;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::NEIGHBOR_ALLREDUCE;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  return status;
}

Status EnqueueTensorWindowPut(std::shared_ptr<Tensor> tensor,
                             const std::string& name,
                             const std::unordered_map<int, float>& dst_weights,
                             const int device, StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::WIN_PUT;
  e.dst_weights = dst_weights;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  return status;
}


Status EnqueueTensorWindowAccumulate(std::shared_ptr<Tensor> tensor,
                             const std::string& name,
                             const std::unordered_map<int, float>& dst_weights,
                             const int device, 
                             const bool require_mutex, 
                             StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::WIN_ACCUMULATE;
  e.dst_weights = dst_weights;
  e.require_mutex = require_mutex;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  return status;
}

Status EnqueueTensorWindowGet(const std::string& name,
                             const std::unordered_map<int, float>& src_weights,
                             StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::WIN_GET;
  e.src_weights = src_weights;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  return status;
}

Status WindowCreate(std::shared_ptr<Tensor> tensor,
                    std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                    const std::string& name, const int device) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinCreate(tensor, neighbor_tensors,
                                                       name, device);
  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot create the MPI_Win for " << name;
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowSync(const std::string& name, int device) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinSync(name, device);
  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot sync the MPI_Win for " << name;
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowFree(const std::string& name, int device) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status;
  if (name.empty()) {
    status = bluefog_global.controller->WinFreeAll();
  } else {
    status = bluefog_global.controller->WinFree(name, device);
  }
  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot free the MPI_Win for " << name;
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowFence(const std::string& name) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinFence(name);

  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot free the MPI_Win for " << name;
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status Barrier(StatusCallback callback) {
  TensorTableEntry e;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::BARRIER;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  return status;
}

Status WindowLock(const std::string& name) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinLock(name);

  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot Lock the MPI_Win for " << name;
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowUnlock(const std::string& name) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinUnlock(name);

  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot Unlock the MPI_Win for " << name;
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowMutexAcquire(const std::vector<int>& acquire_ranks) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinMutexAcquire(acquire_ranks);

  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot acquire window mutex";
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowMutexRelease(const std::vector<int>& release_ranks) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinMutexRelease(release_ranks);

  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot release window mutex"; 
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status GetBluefogTimeline(Timeline*& timeline) {
  timeline = &(bluefog_global.timeline);
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  if (!bluefog_global.timeline_enabled) {
    return Status::Aborted("timeline is not enabled.");
  }
  return Status::OK();
}

}  // namespace common
}  // namespace bluefog
