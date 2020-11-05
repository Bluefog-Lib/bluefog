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
#include <chrono>
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
#include "message.h"

#if HAVE_NCCL
#include "nccl_controller.h"
#endif

// Bluefog knobs.
#define COORDINATE_RANK 0
#define BLUEFOG_TIMELINE "BLUEFOG_TIMELINE"
#define BLUEFOG_CYCLE_TIME "BLUEFOG_CYCLE_TIME"
#define BLUEFOG_FUSION_THRESHOLD "BLUEFOG_FUSION_THRESHOLD"

// Stall-check warning time
#define STALL_WARNING_TIME std::chrono::seconds(60)

namespace bluefog {
namespace common {

namespace {

// All the Bluefog state that must be stored globally per-process.
BluefogGlobalState bluefog_global;

MPIContext mpi_context;

#if HAVE_NCCL
NCCLContext nccl_context;
#endif

// If set, win_ops will execute the same ops on associated p as well.
static bool global_with_associated_p_state = false;
static bool global_skip_negotiate_stage = false;

const auto SUSPEND_BACKGROUND_WAITTING_DURATION = std::chrono::microseconds(10);

// Table for storing Tensor metadata on rank zero. This is used for error
// checking, stall checking and size calculations, as well as determining
// when a reduction is ready to be done (when all nodes are ready to do it).
using MessageTable = std::unordered_map<
    std::string,
    std::tuple<std::vector<Request>, std::chrono::steady_clock::time_point>>;

// Store the Request for a name, and return whether the total count of
// Requests for that tensor is now equal to the MPI size (and thus we are
// ready to reduce the tensor).
bool IncrementTensorCount(MessageTable* message_table, const Request& msg,
                          int mpi_size) {
  std::string name = msg.tensor_name();
  auto table_iter = message_table->find(name);
  if (table_iter == message_table->end()) {
    std::vector<Request> messages = {msg};
    messages.reserve(static_cast<unsigned long>(mpi_size));

    auto now = std::chrono::steady_clock::now();
    message_table->emplace(name, std::make_tuple(std::move(messages), now));
    // ready_to_reduce unless mpi_size is 1 because only 1 message is inserted.
    return mpi_size == 1;
  }

  std::vector<Request>& messages = std::get<0>(table_iter->second);
  messages.push_back(msg);
  int count = (int)messages.size();
  bool ready_to_reduce = count == mpi_size;
  return ready_to_reduce;
}

bool CheckRequestAndDataType(const std::vector<Request>& requests,
                             std::ostringstream& error_message_stream) {
  bool error = false;
  // Check that all requested operations are the same
  auto message_type = requests[0].request_type();
  assert(message_type == Request::ALLREDUCE ||
         message_type == Request::BROADCAST ||
         message_type == Request::ALLGATHER ||
         message_type == Request::NEIGHBOR_ALLGATHER ||
         message_type == Request::NEIGHBOR_ALLREDUCE ||
         message_type == Request::WIN_CREATE ||
         message_type == Request::WIN_FREE);
  for (unsigned int i = 1; i < requests.size(); i++) {
    auto request_type = requests[i].request_type();
    if (message_type != request_type) {
      error = true;
      error_message_stream << "Mismatched MPI operations: One rank did an "
                           << Request::RequestType_Name(message_type)
                           << ", but another rank did an "
                           << Request::RequestType_Name(request_type) << ".";
      break;
    }
  }

  // Check that all data types of tensors are identical.
  // WIN_FREE doesn't need that since win_create have checked the size for
  // each associated name is the same. Also, win_free does not provid size
  // information since it can be WinFreeAll.
  if (message_type != Request::WIN_FREE) {
    auto data_type = requests[0].tensor_type();
    for (unsigned int i = 1; i < requests.size(); i++) {
      auto request_type = requests[i].tensor_type();
      if (data_type != request_type) {
        error = true;
        error_message_stream << "Mismatched data types: One rank had type "
                             << DataType_Name(data_type)
                             << ", but another rank had type "
                             << DataType_Name(request_type) << ".";
        break;
      }
    }
  }
  return error;
}

bool CheckRequestRootRank(const std::vector<Request>& requests,
                          std::ostringstream& error_message_stream) {
  auto message_type = requests[0].request_type();
  bool error = false;
  int first_root_rank = requests[0].root_rank();
  for (unsigned int i = 1; i < requests.size(); i++) {
    int this_root_rank = requests[i].root_rank();
    if (first_root_rank != this_root_rank) {
      error = true;
      error_message_stream << "Mismatched "
                           << Request::RequestType_Name(message_type)
                           << " root ranks: One rank specified root rank "
                           << first_root_rank
                           << ", but another rank specified root rank "
                           << this_root_rank << ".";
      break;
    }
  }
  return error;
}

bool CheckRequestIsHierarchical(const std::vector<Request>& requests,
                                std::ostringstream& error_message_stream) {
  auto message_type = requests[0].request_type();
  bool error = false;
  bool first_is_hierarchical = requests[0].is_hierarchical();
  for (unsigned int i = 1; i < requests.size(); i++) {
    bool this_is_hierarchical = requests[i].is_hierarchical();
    if (first_is_hierarchical != this_is_hierarchical) {
      error = true;
      error_message_stream
          << "Mismatched " << Request::RequestType_Name(message_type)
          << " is_hierarchical ops. Some ranks specified for hierarchical ops "
          << " but some ranks are not.";
      break;
    }
  }
  return error;
}

bool CheckRequestTensorShape(const std::vector<Request>& requests,
                             std::ostringstream& error_message_stream) {
  bool error = false;
  auto message_type = requests[0].request_type();
  TensorShape tensor_shape;
  for (auto dim : requests[0].tensor_shape()) {
    tensor_shape.AddDim(dim);
  }
  for (unsigned int i = 1; i < requests.size(); i++) {
    TensorShape request_shape;
    for (auto dim : requests[i].tensor_shape()) {
      request_shape.AddDim(dim);
    }
    if (tensor_shape != request_shape) {
      error = true;
      error_message_stream << "Mismatched "
                           << Request::RequestType_Name(message_type)
                           << " tensor shapes: One rank sent a tensor of shape "
                           << tensor_shape.DebugString()
                           << ", but another rank sent a tensor of shape "
                           << request_shape.DebugString() << ".";
      break;
    }
  }
  return error;
}

bool CheckRequestGatherTensorShape(const std::vector<Request>& requests,
                                   std::ostringstream& error_message_stream) {
  auto message_type = requests[0].request_type();
  bool error = false;
  TensorShape tensor_shape;
  for (auto dim : requests[0].tensor_shape()) {
    tensor_shape.AddDim(dim);
  }

  if (tensor_shape.dims() == 0) {
    error = true;
    error_message_stream << "Rank zero tried to "
                         << Request::RequestType_Name(message_type)
                         << " a rank-zero tensor.";
  }

  for (unsigned int i = 1; i < requests.size(); i++) {
    TensorShape request_shape;
    for (auto dim : requests[i].tensor_shape()) {
      request_shape.AddDim(dim);
    }
    if (tensor_shape.dims() != request_shape.dims()) {
      error = true;
      error_message_stream << "Mismatched "
                           << Request::RequestType_Name(message_type)
                           << " tensor shapes: One rank sent a tensor of rank "
                           << tensor_shape.dims()
                           << ", but another rank sent a tensor of rank "
                           << request_shape.dims() << ".";
      break;
    }

    bool dim_mismatch = false;
    for (int dim = 1; dim < tensor_shape.dims(); dim++) {
      if (tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor with dimension " << dim
            << " equal to " << tensor_shape.dim_size(dim)
            << ", but another rank sent a tensor with dimension " << dim
            << " equal to " << request_shape.dim_size(dim) << ".";
        dim_mismatch = true;
        break;
      }
    }
    if (dim_mismatch) {
      break;
    }
  }
  return error;
}

bool CheckRequestDevice(const std::vector<Request>& requests,
                        std::ostringstream& error_message_stream) {
  auto message_type = requests[0].request_type();
  bool error = false;
  bool first_device_is_cpu = requests[0].device() == CPU_DEVICE_ID;
  for (unsigned int i = 1; i < requests.size(); i++) {
    bool this_device_is_cpu = requests[i].device() == CPU_DEVICE_ID;
    if (first_device_is_cpu != this_device_is_cpu) {
      error = true;
      error_message_stream
          << "Mismatched " << Request::RequestType_Name(message_type)
          << " CPU/GPU device selection: One rank specified device "
          << (first_device_is_cpu ? "CPU" : "GPU")
          << ", but another rank specified device "
          << (this_device_is_cpu ? "CPU" : "GPU") << ".";
      break;
    }
  }
  return error;
}

// Once a tensor is ready to be reduced, the coordinator sends an MPIResponse
// instructing all ranks to start the reduction to all ranks. The MPIResponse
// also contains error messages in case the submitted MPIRequests were not
// valid (for example, contained mismatched shapes or types).
//
// Constructing the MPIResponse, thus, requires a whole lot of error checking.
Response ConstructResponse(MessageTable* message_table, std::string name) {
  bool error = false;
  auto it = message_table->find(name);
  assert(it != message_table->end());

  std::vector<Request>& requests = std::get<0>(it->second);
  assert(requests.size() > 0);
  auto message_type = requests[0].request_type();

  std::ostringstream error_message_stream;

  // Make sure all requests' message type and data type are the same.
  error = CheckRequestAndDataType(requests, error_message_stream);

  // Make sure all requests use the same device.
  if (!error) {
    error = CheckRequestDevice(requests, error_message_stream);
  }

  // If we are doing a broadcast, check that all root ranks are identical.
  if (!error) {
    if (message_type == Request::BROADCAST) {
      error = CheckRequestRootRank(requests, error_message_stream);
    }
  }

  // If we are doing allreduce, make sure all are Hierarchical or are all not.
  if (!error) {
    if (message_type == Request::ALLREDUCE ||
        message_type == Request::NEIGHBOR_ALLREDUCE) {
      error = CheckRequestIsHierarchical(requests, error_message_stream);
    }
  }

  // If we are doing an (neighbor_)allreduce or broadcast, check that all tensor
  // shapes are identical.
  if (!error) {
    if (message_type == Request::ALLREDUCE ||
        message_type == Request::BROADCAST ||
        message_type == Request::NEIGHBOR_ALLREDUCE ||
        message_type == Request::WIN_CREATE) {
      error = CheckRequestTensorShape(requests, error_message_stream);
    }
  }

  // If we are doing (neighbor_)allgather, make sure all but the first dimension
  // are the same. The first dimension may be different and the output tensor is
  // the sum of the first dimension.
  if (!error) {
    if (message_type == Request::ALLGATHER ||
        message_type == Request::NEIGHBOR_ALLGATHER) {
      error = CheckRequestGatherTensorShape(requests, error_message_stream);
    }
  }

  std::vector<int32_t> devices;
  if (!error) {
    devices.resize(requests.size());
    for (auto& request : requests) {
      devices[request.request_rank()] = request.device();
    }
  }

  Response response;
  response.add_tensor_name(name);
  if (error) {
    std::string error_message = error_message_stream.str();
    response.set_response_type(Response::ERROR);
    response.set_error_message(error_message);
  } else if (message_type == Request::ALLGATHER) {
    response.set_response_type(Response::ALLGATHER);
  } else if (message_type == Request::ALLREDUCE) {
    response.set_response_type(Response::ALLREDUCE);
  } else if (message_type == Request::BROADCAST) {
    response.set_response_type(Response::BROADCAST);
  } else if (message_type == Request::NEIGHBOR_ALLGATHER) {
    response.set_response_type(Response::NEIGHBOR_ALLGATHER);
  } else if (message_type == Request::NEIGHBOR_ALLREDUCE) {
    response.set_response_type(Response::NEIGHBOR_ALLREDUCE);
  } else if (message_type == Request::WIN_CREATE) {
    response.set_response_type(Response::WIN_CREATE);
  } else if (message_type == Request::WIN_FREE) {
    response.set_response_type(Response::WIN_FREE);
  }
  response.set_devices(devices);

  // Clear all queued up requests for this name. They are now taken care of
  // by the constructed MPI response.
  message_table->erase(it);

  return response;
}

// Report Tensors that were submitted to be reduced, gathered or broadcasted by
// some ranks but not others and are waiting for long time to get processed.
void CheckForStalledTensors(BluefogGlobalState& state) {
  bool preamble = false;
  auto now = std::chrono::steady_clock::now();
  for (auto& m : *state.message_table) {
    auto tensor_name = m.first;
    std::vector<Request>& messages = std::get<0>(m.second);
    std::chrono::steady_clock::time_point start_at = std::get<1>(m.second);

    if (now - start_at > STALL_WARNING_TIME) {
      if (!preamble) {
        std::cerr << "WARNING: One or more tensors were submitted to be "
                     "reduced, gathered or broadcasted by subset of ranks and "
                     "are waiting for remainder of ranks for more than "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         STALL_WARNING_TIME)
                         .count()
                  << " seconds. ";
        std::cerr << "This may indicate that different ranks are trying to "
                     "submit different tensors or that only subset of ranks is "
                     "submitting tensors, which will cause deadlock. " << std::endl;
        std::cerr << "Stalled ops:" << std::endl;
        preamble = true;
      }
      std::cerr << tensor_name;
      std::cerr << " [missing ranks:";
      std::unordered_set<int32_t> ready_ranks;
      bool missing_preamble = false;
      for (auto msg_iter = messages.begin(); msg_iter != messages.end();
           msg_iter++) {
        ready_ranks.insert(msg_iter->request_rank());
      }
      for (int32_t rank = 0; rank < mpi_context.size_; rank++) {
        if (ready_ranks.find(rank) == ready_ranks.end()) {
          if (!missing_preamble) {
            std::cerr << " ";
            missing_preamble = true;
          } else {
            std::cerr << ", ";
          }
          std::cerr << rank;
        }
      }
      std::cerr << "]" << std::endl;
    }
  }
}

}  // namespace

bool RunLoopOnce(BluefogGlobalState& state);

void BackgroundThreadLoop(BluefogGlobalState& state) {
  auto mpi_ctx_manager = MPIContextManager();
  mpi_context.Initialize(std::vector<int>{}, mpi_ctx_manager);

  // Initialize controller
  state.controller->Initialize();

  // We use Lazy initialized pattern. nccl_controller will be initialized only
  // when it is necessary.

  // Open the timeline file
  char* bluefog_timeline_loc = std::getenv(BLUEFOG_TIMELINE);
  if (bluefog_timeline_loc != nullptr) {
    std::string timeline_filename = std::string(bluefog_timeline_loc) +
                                    std::to_string(mpi_context.rank_) +
                                    std::string(".json");
    state.timeline.Initialize(timeline_filename, mpi_context.size_);
    state.timeline_enabled = true;
    BFLOG(TRACE, mpi_context.rank_)
        << "timeline " << timeline_filename << " init done";
  }
  // Override the cycle time.
  char* bluefog_cycle_time = std::getenv(BLUEFOG_CYCLE_TIME);
  if (bluefog_cycle_time != nullptr) {
    state.cycle_time_ms = std::strtof(bluefog_cycle_time, nullptr);
  }

  // Override Tensor Fusion threshold, if it's set.
  auto bluefog_fusion_threshold = std::getenv(BLUEFOG_FUSION_THRESHOLD);
  if (bluefog_fusion_threshold != nullptr) {
    state.tensor_fusion_threshold =
        std::strtol(bluefog_fusion_threshold, nullptr, 10);
  }

  // Initialize the tensor count table. No tensors are available yet.
  if (bluefog_global.controller->GetRank() == COORDINATE_RANK) {
    state.message_table = std::unique_ptr<MessageTable>(new MessageTable());
  }

  // Signal that initialization is completed.
  state.initialization_done = true;
  BFLOG(INFO, bluefog_global.controller->GetRank()) << "Bluefog Initialized";

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
#if HAVE_NCCL
  // NCCL context has to be finalized before MPI since it relied on
  // several functions of MPI.
  if (nccl_context.is_initialized) {
    nccl_context.Finalize();
  }
#endif
  mpi_context.Finalize(mpi_ctx_manager);
}

Vendor DetermineController(const MPIOpsType& op_type, int device) {
  if (device == CPU_DEVICE_ID) return Vendor::MPI;

  bool nccl_impl_available = true;
  bool force_mpi = false;
  bool built_with_nccl = false;
#if HAVE_NCCL
  built_with_nccl = true;
#endif
  char* by_mpi_env;
  switch (op_type) {
    case MPIOpsType::ALLREDUCE:
      by_mpi_env = std::getenv("BLUEFOG_ALLREDUCE_BY_MPI");
      break;
    case MPIOpsType::BROADCAST:
      by_mpi_env = std::getenv("BLUEFOG_BROADCAST_BY_MPI");
      break;
    case MPIOpsType::ALLGATHER:
      by_mpi_env = std::getenv("BLUEFOG_ALLGATHER_BY_MPI");
      break;
    case MPIOpsType::NEIGHBOR_ALLGATHER:
      by_mpi_env = std::getenv("BLUEFOG_NEIGHBOR_ALLGATHER_BY_MPI");
      break;
    case MPIOpsType::NEIGHBOR_ALLREDUCE:
      by_mpi_env = std::getenv("BLUEFOG_NEIGHBOR_ALLREDUCE_BY_MPI");
      break;
    case MPIOpsType::WIN_PUT:
    case MPIOpsType::WIN_GET:
    case MPIOpsType::WIN_ACCUMULATE:
    case MPIOpsType::WIN_CREATE:
    case MPIOpsType::WIN_FREE:
    case MPIOpsType::WIN_SYNC:
      by_mpi_env = std::getenv("BLUEFOG_WIN_OPS_BY_MPI");
      break;
    default:
      by_mpi_env = nullptr;
      nccl_impl_available = false;
  }
  force_mpi = (by_mpi_env != nullptr) && (*by_mpi_env == '1');
  if (!built_with_nccl || !nccl_impl_available || force_mpi) return Vendor::MPI;
  return Vendor::NCCL;
}

void PerformOperation(std::vector<TensorTableEntry>& entries) {
  auto& timeline = bluefog_global.timeline;
  for (auto& entry : entries) {
    Vendor controller_vendor =
        DetermineController(entry.mpi_ops_type, entry.device);
#if HAVE_NCCL
    if (controller_vendor == Vendor::NCCL && !nccl_context.is_initialized) {
      bluefog_global.nccl_controller->Initialize();
      BFLOG(INFO, bluefog_global.controller->GetRank()) << "NCCL Initialized";
    }
#endif

    // Wait for the data is ready (in GPU case).
    if (entry.ready_event != nullptr) {
      while (!entry.ready_event->Ready()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(100));
      }
    }

    switch (entry.mpi_ops_type) {
      case MPIOpsType::ALLREDUCE:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
        timeline.ActivityStart(entry.tensor_name, "PROC_ALLREDUCE");
        if (controller_vendor == Vendor::MPI) {
          bluefog_global.controller->Allreduce(entry);
        }
#if HAVE_NCCL
        if (controller_vendor == Vendor::NCCL) {
          bluefog_global.nccl_controller->Allreduce(entry);
        }
#endif
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::BROADCAST:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
        timeline.ActivityStart(entry.tensor_name, "PROC_BROADCAST");
        if (controller_vendor == Vendor::MPI) {
          bluefog_global.controller->Broadcast(entry);
        }
#if HAVE_NCCL
        if (controller_vendor == Vendor::NCCL) {
          bluefog_global.nccl_controller->Broadcast(entry);
        }
#endif
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::ALLGATHER:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
        timeline.ActivityStart(entry.tensor_name, "PROC_ALLGATHER");
        if (controller_vendor == Vendor::MPI) {
          bluefog_global.controller->Allgather(entry);
        }
#if HAVE_NCCL
        if (controller_vendor == Vendor::NCCL) {
          bluefog_global.nccl_controller->Allgather(entry);
        }
#endif
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::NEIGHBOR_ALLGATHER:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name  << " with "
            << Vendor_Name(controller_vendor);
        timeline.ActivityStart(entry.tensor_name, "PROC_NEIGHBOR_ALLGATHER");
        if (controller_vendor == Vendor::MPI) {
          bluefog_global.controller->NeighborAllgather(entry);
        }
#if HAVE_NCCL
        if (controller_vendor == Vendor::NCCL) {
          bluefog_global.nccl_controller->NeighborAllgather(entry);
        }
#endif
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::NEIGHBOR_ALLREDUCE:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
        timeline.ActivityStart(entry.tensor_name, "PROC_NEIGHBOR_ALLREDUCE");
        if (controller_vendor == Vendor::MPI) {
          bluefog_global.controller->NeighborAllreduce(entry);
        }
#if HAVE_NCCL
        if (controller_vendor == Vendor::NCCL) {
          bluefog_global.nccl_controller->NeighborAllreduce(entry);
        }
#endif
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::PAIR_GOSSIP:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
        timeline.ActivityStart(entry.tensor_name, "PROC_PAIR_GOSSIP");
        bluefog_global.controller->PairGossip(entry);
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::BARRIER:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing Barrier now ";
        bluefog_global.controller->Barrier(entry);
        break;
      case MPIOpsType::WIN_PUT:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing WIN_PUT on " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
        timeline.ActivityStart(entry.tensor_name, "WIN_PUT");
        if (controller_vendor == Vendor::MPI) {
          bluefog_global.controller->WinPut(entry);
        }
#if HAVE_NCCL
        if (controller_vendor == Vendor::NCCL) {
          bluefog_global.nccl_controller->WinPut(entry);
        }
#endif
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::WIN_GET:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing WIN_GET on " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
        timeline.ActivityStart(entry.tensor_name, "WIN_GET");
        if (controller_vendor == Vendor::MPI) {
          bluefog_global.controller->WinGet(entry);
        }
#if HAVE_NCCL
        if (controller_vendor == Vendor::NCCL) {
          bluefog_global.nccl_controller->WinGet(entry);
        }
#endif
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::WIN_ACCUMULATE:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing WIN_ACCUMULATE on " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
        timeline.ActivityStart(entry.tensor_name, "WIN_ACCUMULATE");
        if (controller_vendor == Vendor::MPI) {
          bluefog_global.controller->WinAccumulate(entry);
        }
#if HAVE_NCCL
        if (controller_vendor == Vendor::NCCL) {
          bluefog_global.nccl_controller->WinAccumulate(entry);
        }
#endif
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::WIN_CREATE:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing WIN_CREATE " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
#if HAVE_NCCL
        if (controller_vendor == Vendor::NCCL) {
          bluefog_global.nccl_controller->WinCreate(entry);
        }
#endif
        if (controller_vendor == Vendor::MPI) {
          bluefog_global.controller->WinCreate(entry);
        }
        break;
      case MPIOpsType::WIN_FREE:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing WIN_FREE " << entry.tensor_name << " with "
            << Vendor_Name(controller_vendor);
        // If the entry doesn't specify the tensor name, which means it will
        // free all windows. Since the controller vender is determined through
        // the device of operating tensors. Apparently, all windows case may
        // have multiple cases. Hence, WinFreeAll will be called in both 
        // controllers not matter what.
#if HAVE_NCCL
        if (nccl_context.is_initialized && entry.tensor_name.empty()) {
          bluefog_global.nccl_controller->WinFreeAll(entry);
        } else {
          if (controller_vendor == Vendor::NCCL) {
            bluefog_global.nccl_controller->WinFree(entry);
          }
        }
#endif
        if (entry.tensor_name.empty()) {
          bluefog_global.controller->WinFreeAll(entry);
        } else {
          if (controller_vendor == Vendor::MPI) {
            bluefog_global.controller->WinFree(entry);
          }
        }
        break;
      default:
        timeline.ActivityEnd(entry.tensor_name);  // End activity for enqueue
        throw std::runtime_error("Unsupported/Unkown MPI Operation Types");
    }
  }
}

void PerformOperationWithFusion(std::vector<TensorTableEntry>& entries) {
  auto& timeline = bluefog_global.timeline;
  assert(entries.size() > 1);
  auto& first_entry = entries[0];
  Vendor controller_vendor =
      DetermineController(first_entry.mpi_ops_type, first_entry.device);
#if HAVE_NCCL
  if (controller_vendor == Vendor::NCCL && !nccl_context.is_initialized) {
    bluefog_global.nccl_controller->Initialize();
    BFLOG(INFO, bluefog_global.controller->GetRank()) << "NCCL Initialized";
  }
#endif

  Status status = bluefog_global.fusion_buffer.InitializeBuffer(
      bluefog_global.tensor_fusion_threshold, first_entry.device,
      first_entry.context,
      [&]() { timeline.ActivityStartAll(entries, "INIT_FUSION_BUFFER"); },
      [&]() { timeline.ActivityEndAll(entries); });
  if (!status.ok()) {
    for (auto& e : entries) {
      e.callback(status);
    }
    return;
  }

  // Wait for all data are ready.
  for (auto& entry : entries) {
    if (entry.ready_event != nullptr) {
      while (!entry.ready_event->Ready()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(100));
      }
    }
  }

  // Only Allreduce and Neighbor_Allreduce are supported, mainly because
  // other ops either no need to use fusion like win ops or not performance
  // critical ops like allgather, broadcast, etc.
  switch (first_entry.mpi_ops_type) {
    case MPIOpsType::ALLREDUCE:
      BFLOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing fused " << first_entry.tensor_name << " and rest "
          << std::to_string(entries.size()) << " tensors.";
      timeline.ActivityStartAll(entries, "PROC_ALLREDUCE");
      if (controller_vendor == Vendor::MPI) {
        bluefog_global.controller->Allreduce(entries);
      }
#if HAVE_NCCL
      if (controller_vendor == Vendor::NCCL) {
        bluefog_global.nccl_controller->Allreduce(entries);
      }
#endif
      timeline.ActivityEndAll(entries);
      break;
    case MPIOpsType::NEIGHBOR_ALLREDUCE:
      BFLOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing fused " << first_entry.tensor_name << " and rest "
          << std::to_string(entries.size()) << " tensors.";
      timeline.ActivityStartAll(entries, "PROC_NEIGHBOR_ALLREDUCE");
      if (controller_vendor == Vendor::MPI) {
        bluefog_global.controller->NeighborAllreduce(entries);
      }
#if HAVE_NCCL
      if (controller_vendor == Vendor::NCCL) {
        bluefog_global.nccl_controller->NeighborAllreduce(entries);
      }
#endif
      timeline.ActivityEndAll(entries);
      break;
    default:
      throw std::runtime_error(
          "Only allreduce or neighbor_allreduce should be called within "
          "PerformOperationWithFusion");
  }
}

void NegotiateOfRequestOfMaster(BluefogGlobalState& state,
                                std::deque<Request>& message_queue_buffer,
                                bool& should_change_topo,
                                bool& should_shut_down) {
  std::vector<std::string> ready_to_reduce;
  RequestList message_list;
  message_list.set_shutdown(should_shut_down);
  message_list.set_change_topo(should_change_topo);
  while (!message_queue_buffer.empty()) {
    Request& message = message_queue_buffer.front();
    message_list.add_request(message);
    bool reduce = IncrementTensorCount(state.message_table.get(), message,
                                       mpi_context.size_);
    if (reduce) {
      ready_to_reduce.push_back(message.tensor_name());
    }
    message_queue_buffer.pop_front();
  }

  // Rank zero has put all its own tensors in the tensor count table.
  // Now, it should count all the tensors that are coming from other
  // ranks at this tick.
  // 1. Get message lengths from every rank.
  auto recvcounts = new int[bluefog_size()];
  recvcounts[0] = 0;
  MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, recvcounts, 1, MPI_INT, COORDINATE_RANK,
             mpi_context.mpi_comm);

  // 2. Compute displacements.
  auto displcmnts = new int[bluefog_size()];
  size_t total_size = 0;
  for (int i = 0; i < bluefog_size(); i++) {
    if (i == 0) {
      displcmnts[i] = 0;
    } else {
      displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
    }
    total_size += recvcounts[i];
  }

  // 3. Collect messages from every rank.
  auto buffer = new uint8_t[total_size];
  MPI_Gatherv(nullptr, 0, MPI_BYTE, buffer, recvcounts, displcmnts, MPI_BYTE,
              COORDINATE_RANK, mpi_context.mpi_comm);

  // 4. Process messages.
  for (int i = 1; i < bluefog_size(); i++) {
    auto rank_buffer_ptr = buffer + displcmnts[i];
    RequestList received_message_list;
    RequestList::ParseFromBytes(received_message_list, rank_buffer_ptr);
    for (auto& received_message : received_message_list.requests()) {
      auto& received_name = received_message.tensor_name();
      bool reduce = IncrementTensorCount(state.message_table.get(),
                                         received_message, mpi_context.size_);
      if (reduce) {
        ready_to_reduce.push_back(received_name);
      }
    }
    if (received_message_list.shutdown()) {
      // Received SHUTDOWN request from one of the workers.
      should_shut_down = true;
    }
    if (received_message_list.change_topo()) {
      should_change_topo = true;
    }
  }
  // 5. Free buffers.
  delete[] recvcounts;
  delete[] displcmnts;
  delete[] buffer;

  // At this point, rank zero should have a fully updated tensor count
  // table and should know all the tensors that need to be reduced or
  // gathered, and everyone else should have sent all their information
  // to rank zero. We can now do reductions and gathers; rank zero will
  // choose which ones and in what order, and will notify the other ranks
  // before doing each reduction.
  std::deque<Response> responses;
  for (auto& tensor_name : ready_to_reduce) {
    Response response =
        ConstructResponse(state.message_table.get(), tensor_name);
    responses.push_back(std::move(response));
  }

  ResponseList response_list;
  response_list.set_shutdown(should_shut_down);
  response_list.set_change_topo(should_change_topo);

  while (!responses.empty()) {
    Response response = responses.front();
    assert(response.tensor_names().size() == 1);
    responses.pop_front();

    if (response.response_type() == Response::ResponseType::ALLREDUCE) {
      // Attempt to add more responses to this fused response.
      const TensorTableEntry& entry =
          state.tensor_queue.GetTensorEntry(response.tensor_names()[0]);
      int64_t tensor_size = entry.tensor->size();
      while (!responses.empty()) {
        Response new_response = responses.front();
        assert(new_response.tensor_names().size() == 1);
        const TensorTableEntry& new_entry =
            state.tensor_queue.GetTensorEntry(new_response.tensor_names()[0]);
        int64_t new_tensor_size = new_entry.tensor->size();
        if (response.response_type() == new_response.response_type() &&
            response.devices() == new_response.devices() &&
            entry.tensor->dtype() == new_entry.tensor->dtype() &&
            entry.is_hierarchical == new_entry.is_hierarchical &&
            tensor_size + new_tensor_size <= state.tensor_fusion_threshold) {
          // These tensors will fuse together well.
          tensor_size += new_tensor_size;
          response.add_tensor_name(new_response.tensor_names()[0]);
          responses.pop_front();
        } else {
          // Don't try to fuse additional tensors since they are usually
          // computed in order of requests and skipping tensors may mean
          // that the batch will have to wait longer while skipped tensors
          // could be reduced at that time.
          break;
        }
      }
    } else if (response.response_type() ==
               Response::ResponseType::NEIGHBOR_ALLREDUCE) {
      // Attempt to add more responses to this fused response.
      const TensorTableEntry& entry =
          state.tensor_queue.GetTensorEntry(response.tensor_names()[0]);
      auto IsSameNeighborList =
          [](std::shared_ptr<std::vector<int>> n1,
             std::shared_ptr<std::vector<int>> n2) -> bool {
        if (n1 == nullptr && n2 == nullptr) return true;
        if (n1 == nullptr || n2 == nullptr) return false;
        if (n1->size() != n2->size()) return false;
        // The order matters as well.
        for (int i = 0; i < n1->size(); i++) {
          if (n1->at(i) != n2->at(i)) {
            return false;
          }
        }
        return true;
      };
      // Recall that send_neighbors is empty or not determines we use partial
      // neighbor allreduce or not.
      int num_recv_neighbors = !entry.dynamic_neighbors_enabled
                                   ? mpi_context.neighbor_indgree_
                                   : entry.recv_neighbors->size();
      // Unlike allreduce, the storage for neighbor_allreduce in fusion buffer
      // is like [t_1, t_2 | t_1_n1, t_2_n1, t_1_n2, t_2_n2].
      // Here t_1 and t_2  means self tensor 1 and 2 and _n1 and _n2 means the
      // recieving tensors for neighbor 1 and 2;
      int64_t tensor_size = entry.tensor->size() * (1 + num_recv_neighbors);

      while (!responses.empty()) {
        Response new_response = responses.front();
        assert(new_response.tensor_names().size() == 1);
        const TensorTableEntry& new_entry =
            state.tensor_queue.GetTensorEntry(new_response.tensor_names()[0]);
        int64_t new_tensor_size =
            new_entry.tensor->size() * (1 + num_recv_neighbors);
        if (response.response_type() == new_response.response_type() &&
            response.devices() == new_response.devices() &&
            entry.tensor->dtype() == new_entry.tensor->dtype() &&
            entry.dynamic_neighbors_enabled == new_entry.dynamic_neighbors_enabled &&
            entry.is_hierarchical == new_entry.is_hierarchical &&
            IsSameNeighborList(entry.send_neighbors,
                               new_entry.send_neighbors) &&
            IsSameNeighborList(entry.recv_neighbors,
                               new_entry.recv_neighbors) &&
            tensor_size + new_tensor_size <= state.tensor_fusion_threshold) {
          // These tensors will fuse together well.
          tensor_size += new_tensor_size;
          response.add_tensor_name(new_response.tensor_names()[0]);
          responses.pop_front();
        } else {
          break;
        }
      }
    }

    response_list.add_response(response);
  }

  // Notify all nodes which tensors we'd like to reduce at this step.
  std::string encoded_response;
  ResponseList::SerializeToString(response_list, encoded_response);
  int encoded_response_length = (int)encoded_response.length() + 1;
  MPI_Bcast(&encoded_response_length, 1, MPI_INT, COORDINATE_RANK,
            mpi_context.mpi_comm);
  MPI_Bcast((void*)encoded_response.c_str(), encoded_response_length, MPI_BYTE,
            COORDINATE_RANK, mpi_context.mpi_comm);
  // Perform the collective operation. All nodes should end up performing
  // the same operation.
  for (auto& response : response_list.responses()) {
    std::vector<TensorTableEntry> nego_entries;
    state.tensor_queue.GetTensorEntriesFromResponse(response, nego_entries);
    if (nego_entries.size() > 1) {
      PerformOperationWithFusion(nego_entries);
    } else {
      PerformOperation(nego_entries);
    }
  }

  // Check for stalled tensors.
  if (std::chrono::steady_clock::now() - state.last_stall_check >
      STALL_WARNING_TIME) {
    CheckForStalledTensors(state);
    state.last_stall_check = std::chrono::steady_clock::now();
  }
}

void NegotiateOfRequestOfSlave(BluefogGlobalState& state,
                               std::deque<Request>& message_queue_buffer,
                               bool& should_change_topo,
                               bool& should_shut_down) {
  std::string encoded_message;
  RequestList message_list;
  message_list.set_shutdown(state.shut_down);
  message_list.set_change_topo(should_change_topo);
  while (!message_queue_buffer.empty()) {
    message_list.add_request(message_queue_buffer.front());
    message_queue_buffer.pop_front();
  }
  RequestList::SerializeToString(message_list, encoded_message);
  int encoded_message_length = (int)encoded_message.length() + 1;
  MPI_Gather(&encoded_message_length, 1, MPI_INT, nullptr, 1, MPI_INT,
             COORDINATE_RANK, mpi_context.mpi_comm);
  MPI_Gatherv((void*)encoded_message.c_str(), encoded_message_length, MPI_BYTE,
              nullptr, nullptr, nullptr, MPI_BYTE, COORDINATE_RANK,
              mpi_context.mpi_comm);

  int msg_length;
  MPI_Bcast(&msg_length, 1, MPI_INT, COORDINATE_RANK, mpi_context.mpi_comm);
  auto buffer = new uint8_t[msg_length];
  MPI_Bcast(buffer, msg_length, MPI_BYTE, COORDINATE_RANK,
            mpi_context.mpi_comm);
  ResponseList response_list;
  ResponseList::ParseFromBytes(response_list, buffer);
  delete[] buffer;

  // Perform the collective operation. All nodes should end up performing
  // the same operation.
  for (auto& response : response_list.responses()) {
    std::vector<TensorTableEntry> nego_entries;
    state.tensor_queue.GetTensorEntriesFromResponse(response, nego_entries);
    if (nego_entries.size() > 1) {
      PerformOperationWithFusion(nego_entries);
    } else {
      PerformOperation(nego_entries);
    }
  }

  if (response_list.shutdown()) {
    should_shut_down = true;
  }
  if (response_list.change_topo()) {
    should_change_topo = true;
  }
}

void NegotiationOfRequest(BluefogGlobalState& state,
                          std::deque<Request>& message_queue_buffer,
                          bool& should_change_topo, bool& should_shut_down) {
  if (bluefog_rank() == COORDINATE_RANK) {
    NegotiateOfRequestOfMaster(state, message_queue_buffer, should_change_topo,
                               should_shut_down);
  } else {
    NegotiateOfRequestOfSlave(state, message_queue_buffer, should_change_topo,
                              should_shut_down);
  }
}

bool RunLoopOnce(BluefogGlobalState& state) {
  // The coordinator sends a SHUTDOWN message to trigger shutdown.
  bool should_shut_down = state.shut_down;
  bool should_change_topo = state.setting_topology;

  // This delay determines thread frequency and MPI message latency
  auto sleep_duration =
      state.last_cycle_start +
      std::chrono::microseconds(long(state.cycle_time_ms * 1000.)) -
      std::chrono::steady_clock::now();
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  state.last_cycle_start = std::chrono::steady_clock::now();

  std::deque<Request> message_queue_buffer;
  state.tensor_queue.PopMessagesFromQueue(message_queue_buffer);

  std::vector<TensorTableEntry> entries;
  auto IsRequestConvertToEntryDirectly = [](const Request& request) -> bool {
    return global_skip_negotiate_stage ||
           (request.request_type() != Request::ALLREDUCE &&
            request.request_type() != Request::ALLGATHER &&
            request.request_type() != Request::BROADCAST &&
            request.request_type() != Request::NEIGHBOR_ALLREDUCE &&
            request.request_type() != Request::NEIGHBOR_ALLGATHER &&
            request.request_type() != Request::WIN_CREATE &&
            request.request_type() != Request::WIN_FREE);
  };
  // For these no need to coordinate, put them into entries directly.
  for (auto& request : message_queue_buffer) {
    if (IsRequestConvertToEntryDirectly(request)) {
      entries.push_back(
          state.tensor_queue.GetTensorEntriesFromRequestDirectly(request));
    }
  }
  message_queue_buffer.erase(
      std::remove_if(message_queue_buffer.begin(), message_queue_buffer.end(),
                     IsRequestConvertToEntryDirectly),
      message_queue_buffer.end());

  PerformOperation(entries);

  // For the rest requests, they needs to coordinate and neogiate.
  // Collect all tensors that are ready to be reduced. Record them in the
  // tensor count table (rank zero) or send them to rank zero to be
  // recorded (everyone else).
  if (global_skip_negotiate_stage) {
    // Pass don't do anything.
  } else {
    NegotiationOfRequest(state, message_queue_buffer, should_change_topo,
                         should_shut_down);
  }
  // Seperate the setting topology and negotiate communnication.
  if (should_change_topo) {
    bluefog_global.ready_to_setting_topology = true;
    while (!bluefog_global.setting_topology_done) {
      std::this_thread::sleep_for(SUSPEND_BACKGROUND_WAITTING_DURATION);
    }
    bluefog_global.ready_to_setting_topology = false;
    // Wait for main thread reset.
    while (bluefog_global.setting_topology_done) {
      std::this_thread::sleep_for(SUSPEND_BACKGROUND_WAITTING_DURATION);
    }
  }

  return !should_shut_down;
}

// Start Bluefog background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeBluefogOnce() {
  mpi_context.Enable();  // We always enable mpi since we relied on MPI only now.
  if (!bluefog_global.initialize_flag.test_and_set()) {
    bluefog_global.controller.reset(new MPIController(mpi_context));
#if HAVE_NCCL
    bluefog_global.nccl_controller.reset(
        new NCCLController(nccl_context, mpi_context));
#endif
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
    //bluefog_global.initialize_flag.clear();
    //bluefog_global.shut_down = false;
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
  if (!bluefog_global.controller->IsWinObjectEmpty()) {
    BFLOG(ERROR)
        << "Cannot set the topology because there are window object uncleared.";
    return -1;
  }
#if HAVE_NCCL
  if (!bluefog_global.nccl_controller->IsWinObjectEmpty()) {
    BFLOG(ERROR)
        << "Cannot set the topology because there are window object uncleared.";
    return -1;
  }
#endif
  bluefog_global.setting_topology = true;
  while (!bluefog_global.ready_to_setting_topology.load()) {
    std::this_thread::sleep_for(SUSPEND_BACKGROUND_WAITTING_DURATION);
  }
  bluefog_global.tensor_queue.LockTensorQueue();
  if (bluefog_global.tensor_queue.size() > 0) {
    BFLOG(ERROR)
        << "Cannot set the topology because there are unfinished MPI ops.";
    bluefog_global.tensor_queue.UnlockTensorQueue();
    return -1;
  }

  bool mpi_result = bluefog_global.controller->SetTopology(
      indegree, sources, outdegree, destinations);
#if HAVE_NCCL && NCCL_MINOR < 7
  if (mpi_result && nccl_context.is_initialized) {
    bluefog_global.nccl_controller->DestroyPeerCommunicators();
    bluefog_global.nccl_controller->InitPeerCommunicators();
  }
#endif
  bluefog_global.tensor_queue.UnlockTensorQueue();

  bluefog_global.setting_topology = false;
  bluefog_global.setting_topology_done = true;
  // Wait for the background thread receive the setting_topology_done and
  // close the ready_to_setting_topology epoch.
  while (bluefog_global.ready_to_setting_topology) {
    std::this_thread::sleep_for(SUSPEND_BACKGROUND_WAITTING_DURATION);
  }
  bluefog_global.setting_topology_done = false;
  return mpi_result;
}

int bluefog_set_topology_with_weights(int indegree, const int* sources,
                                      int outdegree, const int* destinations,
                                      double self_weight,
                                      const double* neighbor_weights) {
  int ret = bluefog_set_topology(indegree, sources, outdegree, destinations);
  if (ret != 1) {
    return ret;
  }
  return bluefog_global.controller->SetTopologyWeights(
      indegree, sources, self_weight, neighbor_weights);
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
    double& self_weight_,
    const std::unordered_map<int, double>*& neighbor_weights_) {
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

int bluefog_is_homogeneous() {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->IsHomogeneous();
}

int bluefog_nccl_built() {
  int result = 0;
#if HAVE_NCCL
  result = 1;
  BFLOG(DEBUG) << "NCCL VERSION: " << NCCL_MAJOR << "." << NCCL_MINOR;
#endif
  return result;
}

void bluefog_set_skip_negotiate_stage(bool value) {
  SetSkipNegotiateStageState(value);
}

int bluefog_get_skip_negotiate_stage() {
  return GetSkipNegotiateStageState();
}

}  // extern "C"

Status EnqueueTensorAllreduce(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<OpContext> context,
                              std::shared_ptr<ReadyEvent> ready_event,
                              bool is_hierarchical_local,
                              const std::string& name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_is_hierarchical(is_hierarchical_local);
  message.set_request_type(Request::ALLREDUCE);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.output = output;
  e.device = device;
  e.ready_event = ready_event;
  e.is_hierarchical = is_hierarchical_local;
  e.context = context;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::ALLREDUCE;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorBroadcast(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const int root_rank, const std::string& name,
                              const int device, StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::BROADCAST);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::BROADCAST;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorAllgather(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<OpContext> context,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string& name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::ALLGATHER);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.context = context;
  e.device = device;
  e.ready_event = ready_event;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::ALLGATHER;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorNeighborAllgather(std::shared_ptr<Tensor> tensor,
                                      std::shared_ptr<OpContext> context,
                                      std::shared_ptr<ReadyEvent> ready_event,
                                      const std::string& name, const int device,
                                      StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::NEIGHBOR_ALLGATHER);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.context = context;
  e.device = device;
  e.ready_event = ready_event;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::NEIGHBOR_ALLGATHER;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

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
                                      StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::NEIGHBOR_ALLREDUCE);
  message.set_is_hierarchical(is_hierarchical);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.output = output;
  e.context = context;
  e.ready_event = ready_event;
  e.recv_neighbors = recv_neighbors;
  e.send_neighbors = send_neighbors;
  e.dynamic_neighbors_enabled = dynamic_neighbors_enabled;
  e.is_hierarchical = is_hierarchical;
  e.enable_topo_check = enable_topo_check;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::NEIGHBOR_ALLREDUCE;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorPairGossip(std::shared_ptr<Tensor> tensor,
                               std::shared_ptr<Tensor> output,
                               std::shared_ptr<ReadyEvent> ready_event,
                               const int target_rank, const std::string& name,
                               const int device, StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::PAIR_GOSSIP);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }
  if (!global_skip_negotiate_stage) {
    return Status::InvalidArgument(
        "Currently, pair gossip operation does not support to run under with "
        "negotiate stage setting. Please set skip negotiate stage to be true.");
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = target_rank;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::PAIR_GOSSIP;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorWindowCreate(
    std::shared_ptr<Tensor> tensor,
    std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
    const std::string& name, const int device, StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name("win_create." + name);  // Add prefix to diff win_ops on same window.
  message.set_request_type(Request::WIN_CREATE);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.callback = callback;
  e.device = device;
  e.mpi_ops_type = MPIOpsType::WIN_CREATE;
  e.tensor = tensor;
  e.neighbor_tensors = neighbor_tensors;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorWindowFree(const std::string& name, int device,
                               StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name("win_free." + name);  // Add prefix to diff win_ops on same window.
  message.set_request_type(Request::WIN_FREE);
  message.set_device(device);

  TensorTableEntry e;
  e.tensor_name = name;
  e.callback = callback;
  e.device = device;
  e.mpi_ops_type = MPIOpsType::WIN_FREE;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorWindowPut(std::shared_ptr<Tensor> tensor,
                              const std::string& name,
                              const std::unordered_map<int, double>& dst_weights,
                              const int device, 
                              const bool require_mutex, 
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name("win_put." + name);  // Add prefix to diff win_ops on same window.
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::WIN_PUT);

  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::WIN_PUT;
  e.dst_weights = dst_weights;
  e.win_ops_with_associated_p = global_with_associated_p_state;
  e.require_mutex = require_mutex;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorWindowAccumulate(
    std::shared_ptr<Tensor> tensor, const std::string& name,
    const std::unordered_map<int, double>& dst_weights, const int device,
    const bool require_mutex, StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name("win_accumulate." + name);  // Add prefix to diff win_ops on same window.
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::WIN_ACCUMULATE);

  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::WIN_ACCUMULATE;
  e.dst_weights = dst_weights;
  e.win_ops_with_associated_p = global_with_associated_p_state;
  e.require_mutex = require_mutex;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorWindowGet(const std::string& name,
                              const std::unordered_map<int, double>& src_weights,
                              const int device, const bool require_mutex,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name("win_get." + name);  // Add prefix to diff win_ops on same window.
  message.set_request_type(Request::WIN_GET);

  TensorTableEntry e;
  e.tensor_name = name;
  e.callback = callback;
  e.device = device;
  e.mpi_ops_type = MPIOpsType::WIN_GET;
  e.src_weights = src_weights;
  e.require_mutex = require_mutex;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status ExecuteBarrier(StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = "barrier";
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::BARRIER;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  bluefog_global.controller->Barrier(e);
  return Status::OK();
}

Status WindowSync(const std::string& name, int device) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Vendor vendor = DetermineController(MPIOpsType::WIN_SYNC, device);
  Status status;
#if HAVE_NCCL
  if (vendor == Vendor::NCCL) {
    status = bluefog_global.nccl_controller->WinSync(
        name, device, global_with_associated_p_state);
  }
#endif
  if (vendor == Vendor::MPI) {
    status = bluefog_global.controller->WinSync(name, device,
                                                global_with_associated_p_state);
  }

  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot sync the MPI_Win for " << name;
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowMutexAcquire(const std::string& name,
                          const std::vector<int>& acquire_ranks, int device,
                          bool is_sync) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  // Because mutex is always associated with each win_create ops, it is safe to
  // use the vendor of win_create for window mutex.
  Vendor vendor = DetermineController(MPIOpsType::WIN_CREATE, device);
  Status status;
#if HAVE_NCCL
  if (vendor == Vendor::NCCL) {
    status = bluefog_global.nccl_controller->WinMutexAcquire(
        name, acquire_ranks, is_sync);
  }
#endif
  if (vendor == Vendor::MPI) {
    status = bluefog_global.controller->WinMutexAcquire(name, acquire_ranks,
                                                        is_sync);
  }

  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot acquire window mutex";
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowMutexRelease(const std::string& name,
                          const std::vector<int>& release_ranks, int device,
                          bool is_sync) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  // Because mutex is always associated with each win_create ops, it is safe to
  // use the vendor of win_create for window mutex.
  Vendor vendor = DetermineController(MPIOpsType::WIN_CREATE, device);
  Status status;
#if HAVE_NCCL
  if (vendor == Vendor::NCCL) {
    status = bluefog_global.nccl_controller->WinMutexRelease(
        name, release_ranks, is_sync);
  }
#endif
  if (vendor == Vendor::MPI) {
    status = bluefog_global.controller->WinMutexRelease(name, release_ranks,
                                                        is_sync);
  }

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

Status GetBluefogFusionBuffer(FusionBufferManager*& fusion_buffer) {
  fusion_buffer = &(bluefog_global.fusion_buffer);
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  return Status::OK();
}

// Following ops do not have NCCL support. (Remove them in the future?)
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

Status GetWindowVersion(const std::string& name, std::vector<int>& versions) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }

  Status status =
      bluefog_global.controller->GetWindowVersionValue(name, versions);

  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot get window version";
    BFLOG(ERROR) << status.reason();
  }
  return status;
}

// TODO(ybc) Add NCCL version for this as well.
Status GetWinAssociatedPByNameAndRank(const std::string& name,
                                           const int rank, double* weight) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  return bluefog_global.controller->GetWinAssociatedPByNameAndRank(
      name, rank, weight);
}

Status SetWinAssociatedPByNameAndRank(const std::string& name,
                                           const int rank, const double weight) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  return bluefog_global.controller->SetWinAssociatedPByNameAndRank(
      name, rank, weight);
}

void SetWinOpsWithAssociatedPState(bool value) {
  global_with_associated_p_state = value;
}

bool GetWinOpsWithAssociatedPState() {
  return global_with_associated_p_state;
}

void SetSkipNegotiateStageState(bool value) {
  if (!bluefog_global.initialization_done) {
    BFLOG(ERROR)
        << "Try to set skip negotiate stage before bluefog is initialized.";
    return;
  }
  if (value == global_skip_negotiate_stage) {
    return;
  }
  if (value) {
    // From running negotiate to skipping negotiate, we need to properly turn
    // off negotiate stage. Otherwise, it may hang the processes. Use setting
    // topology flag to suspend the negotiate stage then skip it.
    bluefog_global.setting_topology = true;
    while (!bluefog_global.ready_to_setting_topology.load()) {
      std::this_thread::sleep_for(SUSPEND_BACKGROUND_WAITTING_DURATION);
    }

    global_skip_negotiate_stage = value;

    bluefog_global.setting_topology = false;
    bluefog_global.setting_topology_done = true;
    // Wait for the background thread receive the setting_topology_done and
    // close the ready_to_setting_topology epoch.
    while (bluefog_global.ready_to_setting_topology) {
      std::this_thread::sleep_for(SUSPEND_BACKGROUND_WAITTING_DURATION);
    }
    bluefog_global.setting_topology_done = false;
  } else {
    global_skip_negotiate_stage = value;
  }
}

bool GetSkipNegotiateStageState() {
  return global_skip_negotiate_stage;
}

}  // namespace common
}  // namespace bluefog
