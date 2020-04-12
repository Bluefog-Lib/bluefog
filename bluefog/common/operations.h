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

#ifndef BLUEFOG_COMMON_OPERATIONS_H
#define  BLUEFOG_COMMON_OPERATIONS_H

#include <functional>
#include "common.h"
#include "timeline.h"

namespace bluefog {
namespace common {

// Check that bluefog is initialized.
Status CheckInitialized();

extern "C" {

// C interface to initialize bluefog.
void bluefog_init();

// C interface to shut down bluefog.
void bluefog_shutdown();

// C interface to get index of current bluefog process.
// Returns -1 if bluefog is not initialized.
int bluefog_rank();

// C interface to get index of current bluefog process in the node it is on.
// Returns -1 if bluefog is not initialized.
int bluefog_local_rank();

// C interface to return number of bluefog processes.
// Returns -1 if bluefog is not initialized.
int bluefog_size();

// C interface to return number of bluefog processes in the node it is on.
// Returns -1 if bluefog is not initialized.
int bluefog_local_size();

// C interface to return flag indicating whether MPI multi-threading is
// supported. Returns -1 if Bluefog is not initialized.
int bluefog_mpi_threads_supported();

// C interface to return flag indicating whether MPI_WIN_MODEL is unified or not.
// Returns 1 if model is unified; Returns 0 if model is seperated.
// Returns -1 if Bluefog is not initialized.
int bluefog_unified_mpi_window_model_supported();

// C interface to return number of (in-) neighbor bluefog processes.
// Returns -1 if bluefog is not initialized or topology is not set.
int bluefog_neighbor_size();

// C interface to set the virtual topology for MPI graph communicator.
// Also, the corresponding graph communicator is created.
// Returns -1 if Bluefog is not initialized or failed.
int bluefog_set_topology(int indegree, const int* sources, 
                         int outdegree, const int* destinations);

// C interface to set the virtual topology for MPI graph communicator with
// weights. Also, the corresponding graph communicator is created. Returns -1 if
// Bluefog is not initialized or failed.
int bluefog_set_topology_with_weights(int indegree, const int* sources,
                                      int outdegree, const int* destinations,
                                      float self_weight, const float* neighbor_weights);

// C interface to load the virtual topology for MPI graph communicator.
// Self-rank is never included no matter self-loop is presented in setup or not.
// Returns -1 if Bluefog is not initialized or failed.
int bluefog_load_topology(int* indegree, int*& sources, 
                          int* outdegree, int*& destinations);

// Load the weights for neighbors. 
// TODO(ybc) Make it as C compatible interface.
int bluefog_load_topology_weights(float& self_weight, 
                                  const std::unordered_map<int, float>*& neighbor_weights);


// C interface to allow python to call timeline.
// If start_activity == true, call ActivityStart, else call ActivityEnd.
int bluefog_timeline(const bool start_activity, const char* tensor_name,
                     const char* activity_name);

}

Status EnqueueTensorAllreduce(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              const std::string& name, const int device,
                              StatusCallback callback);

Status EnqueueTensorBroadcast(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              const int root_rank, const std::string& name,
                              const int device, StatusCallback callback);

Status EnqueueTensorAllgather(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<OpContext> context,
                              const std::string& name, const int device,
                              StatusCallback callback);

Status EnqueueTensorNeighborAllgather(std::shared_ptr<Tensor> tensor,
                                      std::shared_ptr<OpContext> context,
                                      const std::string& name, const int device,
                                      StatusCallback callback);

Status EnqueueTensorNeighborAllreduce(std::shared_ptr<OpContext> context,
                                      std::shared_ptr<Tensor> tensor,
                                      std::shared_ptr<Tensor> output,
                                      const std::string& name, const int device,
                                      StatusCallback callback);

Status EnqueueTensorWindowPut(std::shared_ptr<Tensor> tensor,
                             const std::string& name, 
                             const std::unordered_map<int, float>& dst_ranks,
                             const int device,
                             StatusCallback callback);

Status EnqueueTensorWindowAccumulate(
    std::shared_ptr<Tensor> tensor, const std::string& name,
    const std::unordered_map<int, float>& dst_ranks, const int device,
    const bool require_mutex, StatusCallback callback);

Status EnqueueTensorWindowGet(const std::string& name, 
                             const std::unordered_map<int, float>& src_ranks,
                             StatusCallback callback);

Status Barrier(StatusCallback callback);

// Note all following ops are not proccessed through the communication thread.
// it is executed throug the main thread. It may cause some mismatch.

Status WindowCreate(std::shared_ptr<Tensor> tensor,
                    std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                    const std::string& name, int device);

Status WindowSync(const std::string& name, int device);

Status WindowFree(const std::string& name, int device);

Status WindowFence(const std::string& name);

Status WindowLock(const std::string& name);

Status WindowUnlock(const std::string& name);

Status WindowMutexAcquire(const std::vector<int>& acquire_ranks);

Status WindowMutexRelease(const std::vector<int>& release_ranks);

Status GetBluefogTimeline(Timeline*& timeline);

}  // namespace common
}  // namespace bluefog

#endif
