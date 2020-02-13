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
#include "logging.h"
#include "global_state.h"

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
  LOG(INFO, bluefog_global.controller->GetRank()) << "Bluefog Initialized";

  // Iterate until shutdown.
  while (RunLoopOnce(state))
    ;

  LOG(DEBUG, bluefog_global.controller->GetRank()) << "Shutting down background thread";

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
    switch (entry.mpi_ops_type)
    {
    case MPIOpsType::ALLREDUCE:
      LOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing " << entry.tensor_name;
      state.controller->Allreduce(entry);
      break;
    case MPIOpsType::BROADCAST:
      LOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing " << entry.tensor_name;
      state.controller->Broadcast(entry);
      break;
    case MPIOpsType::ALLGATHER:
      LOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing " << entry.tensor_name;
      state.controller->Allgather(entry);
      break;
    case MPIOpsType::NEIGHBOR_ALLGATHER:
      LOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing " << entry.tensor_name;
      state.controller->NeighborAllgather(entry);
      break;
    case MPIOpsType::NEIGHBOR_ALLREDUCE:
      LOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing " << entry.tensor_name;
      state.controller->NeighborAllreduce(entry);
      break;
    case MPIOpsType::WIN_PUT:
      LOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing WIN_PUT on " << entry.tensor_name;
      state.controller->WinPut(entry);
      break;
    case MPIOpsType::WIN_GET:
      LOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing WIN_GET on " << entry.tensor_name;
      state.controller->WinGet(entry);
      break;
    case MPIOpsType::BARRIER:
      LOG(TRACE, bluefog_global.controller->GetRank())
          << "Processing Barrier now ";
      state.controller->Barrier(entry);
      break;
    default:
      throw std::runtime_error("Unsupported/Unkown MPI Operation Types");
    }
  } catch (std::length_error& e) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  } catch (std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return !bluefog_global.shut_down;
}

// Start Bluefog background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeBluefogOnce() {
  mpi_context.Enable();  // We always enable mpi since we relied on MPI only now.
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
  LOG(DEBUG) << "Background thread init done";
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

int bluefog_set_topology(int indegree, const int* sources, int outdegree,
                         const int* destinations) {
  if (!bluefog_global.initialization_done) {
    LOG(ERROR) << "Cannot set the topology because bluefog has not been initialized.";
    return -1;
  }
  if (!bluefog_global.controller->IsWinObjetEmpty()) {
    LOG(ERROR) << "Cannot set the topology because there are window object uncleared.";
    return -1;
  }
  if (bluefog_global.tensor_queue.size() > 0) {
    LOG(ERROR) << "Cannot set the topology because there are unfinished MPI ops.";
    return -1;
  }
  return bluefog_global.controller->SetTopology(indegree, sources, outdegree,
                                                destinations);
}

int bluefog_load_topology(int* indegree, int*& sources, int* outdegree,
                          int*& destinations) {
  if (!bluefog_global.initialization_done) {
    return -1;
  }
  return bluefog_global.controller->LoadTopology(indegree, sources, outdegree,
                                                 destinations);
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
  if (status.ok()) {
    LOG(TRACE, bluefog_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}


Status EnqueueTensorBroadcast(std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              const int root_rank,
                              const std::string& name, const int device,
                              StatusCallback callback) {
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
  if (status.ok()) {
    LOG(TRACE, bluefog_global.controller->GetRank()) << "Enqueued " << name;
  }
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
  if (status.ok()) {
    LOG(TRACE, bluefog_global.controller->GetRank()) << "Enqueued " << name;
  }
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
  if (status.ok()) {
    LOG(TRACE, bluefog_global.controller->GetRank()) << "Enqueued " << name;
  }
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
  if (status.ok()) {
    LOG(TRACE, bluefog_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

Status EnqueuTensorWindowPut(std::shared_ptr<Tensor> tensor,
                             const std::string& name, 
                             const std::vector<int>& dst_ranks,
                             const int device,
                             StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::WIN_PUT;
  e.dst_ranks = dst_ranks;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  if (status.ok()) {
    LOG(TRACE, bluefog_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

Status EnqueuTensorWindowGet(std::shared_ptr<Tensor> tensor,
                             const std::string& name, 
                             const std::vector<int>& src_ranks,
                             const int device,
                             StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.tensor = tensor;
  e.device = device;
  e.callback = callback;
  e.mpi_ops_type = MPIOpsType::WIN_GET;
  e.src_ranks = src_ranks;

  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.tensor_queue.AddToTensorQueue(e);
  if (status.ok()) {
    LOG(TRACE, bluefog_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

Status WindowCreate(std::shared_ptr<Tensor> tensor, 
                    std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                    const std::string& name,
                    const int device) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinCreate(tensor, neighbor_tensors, name, device);
  if (!status.ok()) {
    LOG(ERROR) << "Cannot create the MPI_Win for " << name;
    LOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowSync(const std::string& name) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinSync(name);
  if (!status.ok()) {
    LOG(ERROR) << "Cannot sync the MPI_Win for " << name;
    LOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowFree(const std::string& name) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status;
  if (name.empty()) {
    status = bluefog_global.controller->WinFreeAll();
  } else {
    status = bluefog_global.controller->WinFree(name);
  }
  if (!status.ok()) {
    LOG(ERROR) << "Cannot free the MPI_Win for " << name;
    LOG(ERROR) << status.reason();
  }
  return status;
}

Status WindowFence(const std::string& name) {
  if (bluefog_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = bluefog_global.controller->WinFence(name);

  if (!status.ok()) {
    LOG(ERROR) << "Cannot free the MPI_Win for " << name;
    LOG(ERROR) << status.reason();
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

}  // namespace common
}  // namespace bluefog