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
#include "tensor_queue.h"

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
                                      double self_weight, const double* neighbor_weights);

// C interface to load the virtual topology for MPI graph communicator.
// Self-rank is never included no matter self-loop is presented in setup or not.
// Returns -1 if Bluefog is not initialized or failed.
int bluefog_load_topology(int* indegree, int*& sources, 
                          int* outdegree, int*& destinations);

// Load the weights for neighbors. 
// TODO(ybc) Make it as C compatible interface.
int bluefog_load_topology_weights(double& self_weight, 
                                  const std::unordered_map<int, double>*& neighbor_weights);

// C interface to allow python to call timeline.
// If start_activity == true, call ActivityStart, else call ActivityEnd.
int bluefog_timeline(const bool start_activity, const char* tensor_name,
                     const char* activity_name);

// C interface to return flag indicating if BlueFog is running under homogeneous
// environment or not.
int bluefog_is_homogeneous();

// C interface to return flag indicating if BlueFog was compiled with NCCL support.
int bluefog_nccl_built();

// C interface to set skip negotiate_stage or not.
void bluefog_set_skip_negotiate_stage(bool value);

int bluefog_get_skip_negotiate_stage();

}

Status EnqueueTensorAllreduce(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<OpContext> context,
                              std::shared_ptr<ReadyEvent> ready_event,
                              bool is_hierarchical_local,
                              const std::string& name, const int device,
                              StatusCallback callback);

Status EnqueueTensorBroadcast(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const int root_rank, const std::string& name,
                              const int device, StatusCallback callback);

Status EnqueueTensorAllgather(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<OpContext> context,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string& name, const int device,
                              StatusCallback callback);

Status EnqueueTensorNeighborAllgather(std::shared_ptr<Tensor> tensor,
                                      std::shared_ptr<OpContext> context,
                                      std::shared_ptr<ReadyEvent> ready_event,
                                      const std::string& name, const int device,
                                      StatusCallback callback);

Status EnqueueTensorNeighborAllreduce(std::shared_ptr<Tensor> tensor,
                                      std::shared_ptr<Tensor> output,
                                      std::shared_ptr<OpContext> context,
                                      std::shared_ptr<ReadyEvent> ready_event,
                                      std::shared_ptr<std::vector<int>> recv_neighbors,
                                      std::shared_ptr<std::vector<int>> send_neighbors,
                                      bool dynamic_neighbors_enabled,
                                      bool is_hierarchical,
                                      bool enable_topo_check,
                                      const std::string& name, const int device,
                                      StatusCallback callback);

Status EnqueueTensorPairGossip(std::shared_ptr<Tensor> tensor,
                               std::shared_ptr<Tensor> output,
                               std::shared_ptr<ReadyEvent> ready_event,
                               const int target_rank, const std::string& name,
                               const int device, StatusCallback callback);

Status EnqueueTensorWindowCreate(
    std::shared_ptr<Tensor> tensor,
    std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
    const std::string& name, int device, StatusCallback callback);

Status EnqueueTensorWindowFree(const std::string& name, int device,
                               StatusCallback callback);

Status EnqueueTensorWindowPut(std::shared_ptr<Tensor> tensor,
                              const std::string& name,
                              const std::unordered_map<int, double>& dst_ranks,
                              const int device, const bool require_mutex,
                              StatusCallback callback);

Status EnqueueTensorWindowAccumulate(std::shared_ptr<Tensor> tensor,
                                     const std::string& name,
                                     const std::unordered_map<int, double>& dst_ranks,
                                     const int device, const bool require_mutex,
                                     StatusCallback callback);

Status EnqueueTensorWindowGet(const std::string& name,
                              const std::unordered_map<int, double>& src_ranks,
                              const int device, const bool require_mutex,
                              StatusCallback callback);

// Note all following ops are not proccessed through the communication thread.
// it is executed throug the main thread. It may cause some mismatch.

Status ExecuteBarrier(StatusCallback callback);

Status WindowSync(const std::string& name, int device);

Status WindowMutexAcquire(const std::string& name,
                          const std::vector<int>& acquire_ranks, int device,
                          bool is_sync);

Status WindowMutexRelease(const std::string& name,
                          const std::vector<int>& release_ranks, int device,
                          bool is_sync);

Status GetWinAssociatedPByNameAndRank(const std::string& name, const int rank,
                                      double* weight);

Status SetWinAssociatedPByNameAndRank(const std::string& name, const int rank,
                                      double weight);

Status GetWindowVersion(const std::string& name,
                        std::vector<int>& versions);

void SetWinOpsWithAssociatedPState(bool value);

bool GetWinOpsWithAssociatedPState();

void SetSkipNegotiateStageState(bool value);

bool GetSkipNegotiateStageState();

Status GetBluefogTimeline(Timeline*& timeline);

Status GetBluefogFusionBuffer(FusionBufferManager*& fusion_buffer);

// Following ops do not have NCCL support. (Remove them in the future?)
Status WindowFence(const std::string& name);

Status WindowLock(const std::string& name);

Status WindowUnlock(const std::string& name);


}  // namespace common
}  // namespace bluefog

#endif
