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

}  // namespace

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
  auto& name = msg.tensor_name();
  auto table_iter = message_table->find(name);
  if (table_iter == message_table->end()) {
    std::vector<Request> messages = {msg};
    messages.reserve(static_cast<unsigned long>(mpi_size));

    auto now = std::chrono::steady_clock::now();
    message_table->emplace(name, std::make_tuple(std::move(messages), now));
    table_iter = message_table->find(name);
  } else {
    std::vector<Request>& messages = std::get<0>(table_iter->second);
    messages.push_back(msg);
  }

  std::vector<Request>& messages = std::get<0>(table_iter->second);
  int count = (int)messages.size();
  bool ready_to_reduce = count == mpi_size;
  return ready_to_reduce;
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

  std::ostringstream error_message_stream;

  // Check that all data types of tensors are identical.
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

  // Check that all requested operations are the same
  auto message_type = requests[0].request_type();
  assert(message_type == Request::ALLREDUCE ||
         message_type == Request::BROADCAST ||
         message_type == Request::ALLGATHER ||
         message_type == Request::NEIGHBOR_ALLGATHER ||
         message_type == Request::NEIGHBOR_ALLREDUCE);
  for (unsigned int i = 1; i < requests.size(); i++) {
    if (error) {
      break;
    }

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

  // If we are doing a broadcast, check that all root ranks are identical.
  if (message_type == Request::BROADCAST) {
    int first_root_rank = requests[0].root_rank();
    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      int this_root_rank = requests[i].root_rank();
      if (first_root_rank != this_root_rank) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " root ranks: One rank specified root rank " << first_root_rank
            << ", but another rank specified root rank " << this_root_rank
            << ".";
        break;
      }
    }
  }

  // If we are doing an (neighbor_)allreduce or broadcast, check that all tensor
  // shapes are identical.
  if (message_type == Request::ALLREDUCE ||
      message_type == Request::BROADCAST ||
      message_type == Request::NEIGHBOR_ALLREDUCE) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }
    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape != request_shape) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of shape "
            << tensor_shape.DebugString()
            << ", but another rank sent a tensor of shape "
            << request_shape.DebugString() << ".";
        break;
      }
    }
  }

  // If we are doing (neighbor_)allgather, make sure all but the first dimension
  // are the same. The first dimension may be different and the output tensor is
  // the sum of the first dimension.
  if (message_type == Request::ALLGATHER ||
      message_type == Request::NEIGHBOR_ALLGATHER) {
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
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape.dims() != request_shape.dims()) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
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
  }

  bool first_device_is_cpu = requests[0].device() == CPU_DEVICE_ID;
  for (unsigned int i = 1; i < requests.size(); i++) {
    if (error) {
      break;
    }

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
  std::vector<int32_t> devices(requests.size());
  for (auto& request : requests) {
    devices[request.request_rank()] = request.device();
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
  }
  response.set_devices(devices);

  // Clear all queued up requests for this name. They are now taken care of
  // by the constructed MPI response.
  message_table->erase(it);

  return response;
}

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
                                    std::to_string(bluefog_rank()) +
                                    std::string(".json");
    state.timeline.Initialize(timeline_filename, bluefog_size());
    state.timeline_enabled = true;
  }

  // Override the cycle time.
  char* bluefog_cycle_time = std::getenv(BLUEFOG_CYCLE_TIME);
  if (bluefog_cycle_time != nullptr) {
    state.cycle_time_ms = std::strtof(bluefog_cycle_time, nullptr);
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
      by_mpi_env = std::getenv("BLUEFOG_WIN_OPS_BY_MPI");
      break;
    case MPIOpsType::WIN_GET:
      by_mpi_env = std::getenv("BLUEFOG_WIN_OPS_BY_MPI");
      break;
    case MPIOpsType::WIN_ACCUMULATE:
      by_mpi_env = std::getenv("BLUEFOG_WIN_OPS_BY_MPI");
      break;
    case MPIOpsType::WIN_CREATE:
      by_mpi_env = std::getenv("BLUEFOG_WIN_OPS_BY_MPI");
      break;
    case MPIOpsType::WIN_FREE:
      by_mpi_env = std::getenv("BLUEFOG_WIN_OPS_BY_MPI");
      break;
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
      state.nccl_controller->Initialize();
      BFLOG(INFO, state.controller->GetRank()) << "NCCL Initialized";
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
            << "Processing " << entry.tensor_name;
        timeline.ActivityStart(entry.tensor_name, "ALLREDUCE");
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
            << "Processing " << entry.tensor_name;
        timeline.ActivityStart(entry.tensor_name, "BROADCAST");
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
        timeline.ActivityStart(entry.tensor_name, "ALLGATHER");
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name;
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
        timeline.ActivityStart(entry.tensor_name, "NEIGHBOR_ALLGATHER");
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing " << entry.tensor_name;
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
            << "Processing " << entry.tensor_name;
        timeline.ActivityStart(entry.tensor_name, "NEIGHBOR_ALLREDUCE");
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
            << "Processing " << entry.tensor_name;
        timeline.ActivityStart(entry.tensor_name, "PAIR_GOSSIP");
        bluefog_global.controller->PairGossip(entry);
        timeline.ActivityEnd(entry.tensor_name);
        break;
      case MPIOpsType::BARRIER:
        BFLOG(TRACE, bluefog_global.controller->GetRank())
            << "Processing Barrier now ";
        bluefog_global.controller->Barrier(entry);
        break;
      // TODO(ybc) All above Ops are collective ops. If the order
      // is disarranged, the whole process will hang. This is possible in
      // tensorflow. For example, if two ops are not control dependent to each
      // other, the order of allreduce request by them are undeterminisitc.
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
      default:
        timeline.ActivityEnd(entry.tensor_name);  // End activity for enqueue
        throw std::runtime_error("Unsupported/Unkown MPI Operation Types");
    }
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
  auto IsRequestConvertToEntryDirectly = [](const Request& request) {
    return (request.request_type() != Request::ALLREDUCE &&
            request.request_type() != Request::ALLGATHER &&
            request.request_type() != Request::BROADCAST &&
            request.request_type() != Request::NEIGHBOR_ALLREDUCE &&
            request.request_type() != Request::NEIGHBOR_ALLGATHER);
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

  // For the rest requests, they needs to coordinate and neogiate.
  // Collect all tensors that are ready to be reduced. Record them in the
  // tensor count table (rank zero) or send them to rank zero to be
  // recorded (everyone else).
  std::vector<std::string> ready_to_reduce;
  if (bluefog_rank() == COORDINATE_RANK) {
    RequestList message_list;
    message_list.set_shutdown(should_shut_down);
    message_list.set_change_topo(should_change_topo);
    while (!message_queue_buffer.empty()) {
      Request& message = message_queue_buffer.front(); 
      message_list.add_request(message_queue_buffer.front());
      bool reduce =
          IncrementTensorCount(state.message_table.get(), message, mpi_context.size_);
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
      auto response = responses.front();
      assert(response.tensor_names().size() == 1);
      responses.pop_front();
      // TODO: Tensor Fusion Logics.
      response_list.add_response(response);
    }

    // Notify all nodes which tensors we'd like to reduce at this step.
    std::string encoded_response;
    ResponseList::SerializeToString(response_list, encoded_response);
    int encoded_response_length = (int)encoded_response.length() + 1;
    MPI_Bcast(&encoded_response_length, 1, MPI_INT, COORDINATE_RANK, mpi_context.mpi_comm);
    MPI_Bcast((void*)encoded_response.c_str(), encoded_response_length,
              MPI_BYTE, COORDINATE_RANK, mpi_context.mpi_comm);
    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      state.tensor_queue.GetTensorEntriesFromResponse(response, entries);
      // TODO: tensor fusion logics?
    }
    // TODO: Check for stalled tensors.
  } else {
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
    MPI_Gatherv((void*)encoded_message.c_str(), encoded_message_length,
                MPI_BYTE, nullptr, nullptr, nullptr, MPI_BYTE, COORDINATE_RANK,
                mpi_context.mpi_comm);

    int msg_length;
    MPI_Bcast(&msg_length, 1, MPI_INT, COORDINATE_RANK, mpi_context.mpi_comm);
    auto buffer = new uint8_t[msg_length];
    MPI_Bcast(buffer, msg_length, MPI_BYTE, COORDINATE_RANK, mpi_context.mpi_comm);
    ResponseList response_list;
    ResponseList::ParseFromBytes(response_list, buffer);
    delete[] buffer;

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      state.tensor_queue.GetTensorEntriesFromResponse(response, entries);
      // TODO: tensor fusion logics?
    }

    if (response_list.shutdown()) {
      should_shut_down = true;
    }
    if (response_list.change_topo()) {
      should_change_topo = true;
    }
  }
  // Seperate the setting topology and negotiate communnication.
  if (should_change_topo) {
    state.ready_to_setting_topology = true;
    while (!state.setting_topology_done) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    state.ready_to_setting_topology = false;
  }

  PerformOperation(entries);
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
  bluefog_global.setting_topology_done = false;
  bluefog_global.setting_topology = true;
  while (!bluefog_global.ready_to_setting_topology) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
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
    bluefog_global.nccl_controller->DestroyPeerCommunicator();
    bluefog_global.nccl_controller->InitPeerCommunicator();
  }
#endif
  bluefog_global.tensor_queue.UnlockTensorQueue();

  bluefog_global.setting_topology = false;
  bluefog_global.setting_topology_done = true;
  // Wait for the background thread receive the setting_topology_done and
  // close the ready_to_setting_topology epoch.
  while (bluefog_global.ready_to_setting_topology) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
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

int bluefog_nccl_built() {
  int result = 0;
#if HAVE_NCCL
  result = 1;
  BFLOG(DEBUG) << "NCCL VERSION: " << NCCL_MAJOR << "." << NCCL_MINOR;
#endif
  return result;
}

}  // extern "C"

Status EnqueueTensorAllreduce(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string& name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
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
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::NEIGHBOR_ALLGATHER;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e, message);
  return status;
}

Status EnqueueTensorNeighborAllreduce(std::shared_ptr<OpContext> context,
                                      std::shared_ptr<Tensor> tensor,
                                      std::shared_ptr<Tensor> output,
                                      std::shared_ptr<ReadyEvent> ready_event,
                                      std::shared_ptr<std::vector<int>> recv_neighbors,
                                      std::shared_ptr<std::vector<int>> send_neighbors,
                                      bool enable_topo_check,
                                      const std::string& name, const int device,
                                      StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::NEIGHBOR_ALLREDUCE);
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
                               const int target_rank, const std::string& name,
                               const int device, StatusCallback callback) {
  Request message;
  message.set_request_rank(bluefog_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::PAIR_GOSSIP);

  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = target_rank;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::PAIR_GOSSIP;

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
  message.set_tensor_name(name);
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
  message.set_tensor_name(name);
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
  message.set_tensor_name(name);
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

Status WindowCreate(std::shared_ptr<Tensor> tensor,
                    std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                    const std::string& name, const int device) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Vendor vendor = DetermineController(MPIOpsType::WIN_CREATE, device);
  Status status;
#if HAVE_NCCL
  if (vendor == Vendor::NCCL) {
    if (!nccl_context.is_initialized) {
      bluefog_global.nccl_controller->Initialize();
      BFLOG(INFO, bluefog_global.controller->GetRank()) << "NCCL Initialized";
    }
    status = bluefog_global.nccl_controller->WinCreate(tensor, neighbor_tensors,
                                                       name, device);
  }
#endif
  if (vendor == Vendor::MPI) {
    status = bluefog_global.controller->WinCreate(tensor, neighbor_tensors,
                                                  name, device);
  }

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

Status WindowFree(const std::string& name, int device) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Vendor vendor = DetermineController(MPIOpsType::WIN_FREE, device);
  Status status;
#if HAVE_NCCL
  // No specified name. So both nccl and mpi will Free win.
  if (nccl_context.is_initialized && name.empty()) {
    status = bluefog_global.nccl_controller->WinFreeAll();
  } else {
    if (vendor == Vendor::NCCL) {
      status = bluefog_global.nccl_controller->WinFree(name, device);
    }
  }
#endif
  if (name.empty()) {
    status = bluefog_global.controller->WinFreeAll();
  } else {
    if (vendor == Vendor::MPI) {
      status = bluefog_global.controller->WinFree(name, device);
    }
  }
  if (!status.ok()) {
    BFLOG(ERROR) << "Cannot free the MPI_Win for " << name;
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

}  // namespace common
}  // namespace bluefog
