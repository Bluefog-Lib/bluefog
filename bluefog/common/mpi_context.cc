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

#include <memory>

#include "mpi_context.h"
#include "logging.h"

namespace bluefog {
namespace common {

void MPIContextManager::EnvInitialize(int mpi_threads_required) {
  int mpi_threads_provided;
  MPI_Init_thread(nullptr, nullptr, mpi_threads_required,
                  &mpi_threads_provided);
}

void MPIContextManager::EnvFinalize() {
  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized);
  if (!is_mpi_finalized) {
    MPI_Finalize();
  }
}

void WindowManager::FreeAllWins() {
  for (auto& win_tensor : wins_tensor_vec_) {
    MPI_Win_free(win_tensor.first.get());
  }
  MPI_Win_free(global_win_.get());
  wins_tensor_vec_.clear();
}

MPI_Datatype MPIContext::GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
  return GetMPIDataType(tensor->dtype());
}

MPI_Datatype MPIContext::GetMPIDataType(const DataType dtype) {
  switch (dtype) {
    case DataType::BLUEFOG_UINT8:
      return MPI_UINT8_T;
    case DataType::BLUEFOG_INT8:
      return MPI_INT8_T;
    case DataType::BLUEFOG_UINT16:
      return MPI_UINT16_T;
    case DataType::BLUEFOG_INT16:
      return MPI_INT16_T;
    case DataType::BLUEFOG_INT32:
      return MPI_INT32_T;
    case DataType::BLUEFOG_INT64:
      return MPI_INT64_T;
    case DataType::BLUEFOG_FLOAT32:
      return MPI_FLOAT;
    case DataType::BLUEFOG_FLOAT64:
      return MPI_DOUBLE;
    case DataType::BLUEFOG_BOOL:
      return MPI_C_BOOL;
    case DataType::BLUEFOG_BYTE:
      return MPI_BYTE;
    default:
      throw std::logic_error("Type " + DataType_Name(dtype) +
                             " is not supported in MPI mode.");
  }
}

MPI_Comm MPIContext::GetMPICommunicator(Communicator comm) {
  switch (comm) {
    case Communicator::GLOBAL:
      return mpi_comm;
    case Communicator::LOCAL:
      return local_comm;
    case Communicator::CROSS:
      return cross_comm;
    case Communicator::GRAPH:
      return graph_comm;
    default:
      throw std::logic_error("Communicator " + CommunicatorName(comm) +
                             " is not supported in MPI mode.");
  }
}

int MPIContext::GetMPITypeSize(DataType dtype) {
  int out;
  MPI_Type_size(GetMPIDataType(dtype), &out);
  return out;
}

void MPIContext::Initialize(const std::vector<int>& ranks,
                            MPIContextManager& ctx_manager) {
  if (!enabled_) {
    return;
  }
  // Initialize MPI if it was not initialized. This must happen on the
  // background thread, since not all MPI implementations support being called
  // from multiple threads.
  //
  // In some cases MPI library has multi-threading support, but it slows down
  // certain components, e.g. OpenIB BTL in OpenMPI gets disabled if
  // MPI_THREAD_MULTIPLE is requested.
  //
  // By default, we will ask for multiple threads, so other libraries like
  // mpi4py can be used together with bluefog if multi-threaded MPI is
  // installed.
  int required = MPI_THREAD_MULTIPLE;
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (is_mpi_initialized) {
    int provided;
    MPI_Query_thread(&provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      BFLOG(WARNING)
          << "MPI has already been initialized without "
             "multi-threading support (MPI_THREAD_MULTIPLE). This will "
             "likely cause a segmentation fault.";
    }
  } else {
    // MPI environment has not been created, using manager to initialize.
    ctx_manager.EnvInitialize(required);
    should_finalize = true;
  }

  if (!ranks.empty()) {
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group work_group;
    MPI_Group_incl(world_group, ranks.size(), ranks.data(), &work_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, work_group, 0, &(mpi_comm));
    if (mpi_comm == MPI_COMM_NULL) {
      BFLOG(WARNING) << "Unable to create bluefog communicator, using "
                        "MPI_COMM_WORLD instead.";
      mpi_comm = MPI_COMM_WORLD;
    }
    MPI_Group_free(&world_group);
    MPI_Group_free(&work_group);
  } else if (!mpi_comm) {
    // No ranks were given and no communicator provided to bluefog_init() so use
    // MPI_COMM_WORLD
    BFLOG(DEBUG) << "Using MPI_COMM_WORLD as a communicator.";
    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  } else {
    BFLOG(DEBUG) << "Using the existing mpi_comm.";
  }

  // Create local comm, Determine local rank by querying the local communicator.
  MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm);

  // Get local rank and world rank for cross comm establishment.
  int local_rank, world_rank;
  MPI_Comm_rank(mpi_comm, &world_rank);
  MPI_Comm_rank(local_comm, &local_rank);

  // Create cross node communicator.
  MPI_Comm_split(mpi_comm, local_rank, world_rank, &cross_comm);

  // The real graph communicator creatation is late.
  graph_comm = MPI_COMM_NULL;
  DisableTopoWeights();
}

void MPIContext::Finalize(MPIContextManager& ctx_manager) {
  if (!enabled_) {
    return;
  }
  if (mpi_comm != MPI_COMM_NULL && mpi_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&mpi_comm);
  }

  if (local_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&local_comm);
  }

  if (cross_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&cross_comm);
  }

  if (graph_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&graph_comm);
    DisableTopoWeights();
  }

  UnregisterAllWindowName();

  DestroyWinMutex();

  if (should_finalize) {
    ctx_manager.EnvFinalize();
  }
}

int MPIContext::BuildGraphComm(const int indegree, const int* sources,
                               const int outdegree, const int* destinations) {
  // The weights is about network optimization instead of computation.
  std::vector<int> source_weights(indegree, 1);
  std::vector<int> dest_weights (outdegree, 1);

  int ret_code = MPI_Dist_graph_create_adjacent(
      mpi_comm, indegree, sources, &source_weights[0], outdegree, destinations,
      &dest_weights[0], MPI_INFO_NULL,
      /*reorder=*/0, &graph_comm);

  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "Build distributed graph communicator failed, see MPI output for "
        "details.");
    return -1;
  }
  return 1;
}

bool MPIContext::RegisterWindowName(const std::string& name) {
  if (named_win_map.find(name) != named_win_map.end()) {
    return false;
  }
  auto win_manager_ptr = std::make_shared<WindowManager>();
  named_win_map[name] = win_manager_ptr;
  return true;
}

std::shared_ptr<WindowManager> MPIContext::GetWindowByName(
    const std::string& name) {
  if (named_win_map.find(name) == named_win_map.end()) {
    return nullptr;
  }
  return named_win_map.at(name);
}

bool MPIContext::UnregisterWindowName(const std::string& name) {
  auto it = named_win_map.find(name);
  if (it == named_win_map.end()) {
    return false;
  }
  it->second->FreeAllWins();
  named_win_map.erase(it);
  return true;
}

bool MPIContext::UnregisterAllWindowName() {
  for (auto& kv : named_win_map) {
    kv.second->FreeAllWins();
  }
  named_win_map.clear();
  return true;
}

bool MPIContext::InitializeWinMutex() {
  BFLOG(TRACE) << "InitializeWinMutex is called";
  if (!win_mutex.empty()) {
    return false;
  }

  int self_rank = 0;
  int global_size = 1;
  MPI_Comm_rank(mpi_comm, &self_rank);
  MPI_Comm_size(mpi_comm, &global_size);
  if (global_size <= 1) {
    // We don't need any mutex for this case.
    return false;
  }

  // We only need one value for self mutex.
  self_mutex_mem = std::unique_ptr<int>(new int(0));  // make_unique is c++14 feature.

  int element_size = 0;
  int win_size = 0;
  void* data_buf;

  for (int rank = 0; rank < global_size; rank++) {
    auto mpi_win_ptr = std::make_shared<MPI_Win>();
    if (rank == self_rank) {
      data_buf = self_mutex_mem.get();
      MPI_Type_size(MPI_INT, &element_size);
      win_size = 1 * element_size;
    } else {
      data_buf = nullptr;
      element_size = 1;
      win_size = 0;
    }
    MPI_Win_create(data_buf, win_size, element_size, MPI_INFO_NULL,
                   GetMPICommunicator(Communicator::GLOBAL),
                   mpi_win_ptr.get());
    win_mutex.push_back(mpi_win_ptr);
  }
  return true;
}

bool MPIContext::DestroyWinMutex() {
  BFLOG(TRACE) << "DestroyWinMutex is called";
  if (win_mutex.empty()) {
    return false;
  } 

  for (auto mpi_win_ptr : win_mutex) {
    MPI_Win_free(mpi_win_ptr.get());
  }
  win_mutex.clear();
  self_mutex_mem.reset();
  return true;
}

}  // namespace common
}  // namespace bluefog
