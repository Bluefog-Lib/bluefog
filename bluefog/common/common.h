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

#ifndef BLUEFOG_COMMON_H
#define BLUEFOG_COMMON_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace bluefog {
namespace common {

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

// Device ID used for CPU.
#define CPU_DEVICE_ID (-1)

// List of supported frameworks.
enum class Framework { TENSORFLOW, PYTORCH };

enum class Vendor {
  MPI,
  NCCL,
};

enum class StatusType {
  OK,
  UNKNOWN_ERROR,
  PRECONDITION_ERROR,
  ABORTED,
  INVALID_ARGUMENT,
  IN_PROGRESS
};

enum class DeviceType { CPU, GPU };

enum class Communicator {
  GLOBAL = 0,
  LOCAL = 1,
  CROSS = 2,
  GRAPH = 3,
};

enum class DataType {
  BLUEFOG_UINT8 = 0,
  BLUEFOG_INT8 = 1,
  BLUEFOG_UINT16 = 2,
  BLUEFOG_INT16 = 3,
  BLUEFOG_INT32 = 4,
  BLUEFOG_INT64 = 5,
  BLUEFOG_FLOAT16 = 6,
  BLUEFOG_FLOAT32 = 7,
  BLUEFOG_FLOAT64 = 8,
  BLUEFOG_BOOL = 9,
  BLUEFOG_BYTE = 10,
};

enum class MPIOpsType {
  UNKNOWN = 0,
  ALLREDUCE = 1,
  ALLGATHER = 2,
  BROADCAST = 3,
  NEIGHBOR_ALLREDUCE = 4,
  NEIGHBOR_ALLGATHER = 5,
  BARRIER=9,
  PAIR_GOSSIP = 10,
  WIN_PUT = 6,
  WIN_GET = 7,
  WIN_ACCUMULATE = 8,
  WIN_CREATE = 11,
  WIN_SYNC = 12,
  WIN_FREE = 13,
};

template <typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept {
    return static_cast<typename std::underlying_type<E>::type>(e);
}

inline std::string CommunicatorName(Communicator comm) {
  switch (comm) {
    case Communicator::GLOBAL:
      return "global";
    case Communicator::LOCAL:
      return "local";
    case Communicator::CROSS:
      return "cross";
    case Communicator::GRAPH:
      return "graph";
    default:
      return "<unknown>";
  }
}

const std::string& DataType_Name(DataType value);

const std::string& Vendor_Name(Vendor vendor);

std::size_t DataType_Size(DataType value);

class Status {
 public:
  Status();
  static Status OK();
  static Status UnknownError(std::string message);
  static Status PreconditionError(std::string message);
  static Status Aborted(std::string message);
  static Status InvalidArgument(std::string message);
  static Status InProgress();
  bool ok() const;
  bool in_progress() const;
  StatusType type() const;
  const std::string& reason() const;

 private:
  StatusType type_ = StatusType::OK;
  std::string reason_ = "";
  Status(StatusType type, std::string reason);
};

// Common error status
const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "Bluefog has not been initialized; use bf.init().");

const Status SHUT_DOWN_ERROR = Status::UnknownError(
    "Bluefog has been shut down. This was caused by an exception on one of the "
    "ranks or an attempt to run collective operation like (allreduce, "
    "neighbor_allreduce) or window ops like (win_put, win_create) a tensor "
    "after one of the ranks finished execution. If the shutdown was caused by "
    "an exception, you should see the exception in the log before the first "
    "shutdown message.");

const Status DUPLICATE_NAME_ERROR = Status::InvalidArgument(
    "Requested to collective operation like (allreduce, neighbor_allreduce) or "
    "window ops like (win_put, win_create) a tensor with the same "
    "name as another tensor that is currently being processed.  If you want "
    "to request another tensor, use a different tensor name.");

class TensorShape {
 public:
  void AddDim(int64_t dim);
  void AppendShape(TensorShape& other);

  const std::string DebugString() const;
  int dims() const;
  int64_t dim_size(int idx) const;
  int64_t num_elements() const;
  const std::vector<int64_t>& to_vector() const;

  inline bool operator==(const TensorShape& rhs) const {
    return shape_ == rhs.shape_;
  }

  inline bool operator!=(const TensorShape& rhs) const {
    return shape_ != rhs.shape_;
  }

 private:
  std::vector<int64_t> shape_;
};

class Tensor {
 public:
  virtual const DataType dtype() const = 0;
  virtual const TensorShape shape() const = 0;
  virtual const void* data() const = 0;
  virtual std::shared_ptr<common::Tensor> data_weight(float weight) = 0;
  virtual int64_t size() const = 0;
  virtual ~Tensor() = default;
};

// ReadyEvent is used to inform the CPU when GPU is used.
class ReadyEvent {
public:
  virtual bool Ready() const = 0;
  virtual ~ReadyEvent() = default;
};

class OpContext;

class PersistentBuffer {
 public:
  virtual const void* AccessData(std::shared_ptr<OpContext> context) const = 0;
  virtual ~PersistentBuffer() = default;
};

class OpContext {
 public:
  // These allocators are fully synchronous, unlike TensorFlow counterparts.
  virtual Status AllocatePersistent(
      int64_t size, std::shared_ptr<PersistentBuffer>* tensor) = 0;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) = 0;
  virtual Status AllocateZeros(int64_t num_elements, DataType dtype,
                               std::shared_ptr<Tensor>* tensor) = 0;
  virtual Framework framework() const = 0;
  virtual ~OpContext() = default;
};

// A callback to call after the communication completes. Since the
// allreduce and allgather ops are asynchronous, this callback is what resumes
// computation after the reduction is completed.
using StatusCallback = std::function<void(const Status&)>;

// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the reduction.
struct TensorTableEntry {
  // Name of the tensor.
  std::string tensor_name;
  // Input tensor.
  std::shared_ptr<Tensor> tensor;
  // Used for create window only.
  std::vector<std::shared_ptr<Tensor>> neighbor_tensors;
  // Pre-allocated output tensor.
  std::shared_ptr<Tensor> output;
  // Operation context, used for allocation.
  std::shared_ptr<OpContext> context;
  // Type of MPI operations, such as ALLREDUCE, BROADCAST etc.
  MPIOpsType mpi_ops_type;
  // Root rank for broadcast operation.
  int root_rank = -1;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device = CPU_DEVICE_ID;
  // Event indicating that data is ready.
  std::shared_ptr<ReadyEvent> ready_event;
  // Source and destination of ranks used in win ops.
  // It maps the src(dst) rank to the weight.
  std::unordered_map<int, double> dst_weights = {};
  std::unordered_map<int, double> src_weights = {};

  // Neighbors for dynamic neighbor_allreduce.
  std::shared_ptr<std::vector<int>> send_neighbors;
  std::shared_ptr<std::vector<int>> recv_neighbors;

  // Boolean value if dynamic neighbor is enabled.
  bool dynamic_neighbors_enabled = false;

  // Boolean value for enabling topology check.
  bool enable_topo_check = false;

  // Boolean value for hierarchical operation or not.
  bool is_hierarchical = false;

  // The ops requires the mutex.
  bool require_mutex = false;

  // If set, the associated p window will do same
  // ops as win_ops.
  bool win_ops_with_associated_p = false;

  // A callback to call with the status.
  StatusCallback callback;
};
using TensorTable = std::unordered_map<std::string, TensorTableEntry>;

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_H
