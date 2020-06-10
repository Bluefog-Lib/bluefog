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

#include "nccl_controller.h"

#include "common.h"
#include "cuda_util.h"
#include "timeline.h"
#include "operations.h"

namespace bluefog {
namespace common {

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
    case DataType::BLUEFOG_INT32:
      return ncclInt32;
    case DataType::BLUEFOG_INT64:
      return ncclInt64;
    case DataType::BLUEFOG_FLOAT16:
      return ncclFloat16;
    case DataType::BLUEFOG_FLOAT32:
      return ncclFloat32;
    case DataType::BLUEFOG_FLOAT64:
      return ncclFloat64;
    default:
      throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                             " is not supported in NCCL mode.");
  }
}

void NCCLContext::Initialize(const int rank, const int size,
                             const int local_rank) {
  if (is_initialized) {
    BFLOG(DEBUG)
        << "NCCL context has been initialized but NCCLContext::Initialize "
        << "is called again";
    return;
  }

  // A single rank will create a unique ID and send it to all other ranks to
  // make sure everyone has it.
  ncclUniqueId nccl_id;
  if (rank == 0) ncclGetUniqueId(&nccl_id);
  MPICHECK(
      MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Assume one device per process
  int nDevices = 0;
  CUDACHECK(cudaGetDeviceCount(&nDevices));
  CUDACHECK(cudaSetDevice(local_rank%nDevices));
  CUDACHECK(cudaStreamCreate(&stream));
  NCCLCHECK(ncclCommInitRank(&nccl_comm, size, nccl_id, rank));

  is_initialized = true;
  cuda_device = local_rank;
  return;
}

void NCCLContext::Finalize() {
  NCCLCHECK(ncclCommDestroy(nccl_comm));
  is_initialized = false;
  cuda_device = -1;
}

void NCCLController::Initialize() {
  nccl_ctx_.Initialize(mpi_ctx_.rank_, mpi_ctx_.size_, mpi_ctx_.local_rank_);
}

bool CheckSameRecvSize(const int* recvcounts, const int size) {
  int first_recv_count;
  for (int i = 0; i < size; i++) {
    if (i == 0) first_recv_count = recvcounts[0];
    if (recvcounts[i] != first_recv_count) return false;
  }
  return true;
}

void NCCLController::Allgather(TensorTableEntry& entry) {
  // Unlike MPI allgatherv, which supports the allgather with different sizes,
  // NCCL require all to be the same size. We just use same routine to find the
  // recvcounts and displacement but displacement is not used.
  int* recvcounts = new int[mpi_ctx_.size_];
  int* displcmnts = new int[mpi_ctx_.size_];
  mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GLOBAL);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GLOBAL);
  if (!CheckSameRecvSize(recvcounts, mpi_ctx_.size_)) {
    throw std::runtime_error(
        "ncclAllGather doesn't support varying lenght of vector. Please make "
        "sure the size of tensors is the same among all processes.");
  }

  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();
  void* buffer_data = (void*)entry.output->data();

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  int ret_code = ncclAllGather(sendbuf, buffer_data, num_elements,
                               GetNCCLDataType(entry.output),
                               nccl_ctx_.nccl_comm, nccl_ctx_.stream);
  if (ret_code != ncclSuccess) {
    throw std::runtime_error(
        "ncclAllGather failed, see NCCL output (NCCL_DEBUG=INFO) for details.");
  }
  // completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(nccl_ctx_.stream));

  delete[] recvcounts;
  delete[] displcmnts;

  entry.callback(Status::OK());
}

void NCCLController::Allreduce(TensorTableEntry& entry) {
  const void* sendbuf = entry.tensor->data();
  void* buffer_data = (void*)entry.output->data();
  int num_elements = entry.tensor->shape().num_elements();

  with_device device_guard(entry.device);
  int ret_code = ncclAllReduce(sendbuf, buffer_data, num_elements,
                               GetNCCLDataType(entry.tensor), ncclSum,
                               nccl_ctx_.nccl_comm, nccl_ctx_.stream);
  if (ret_code != ncclSuccess) {
    throw std::runtime_error(
        "ncclAllReduce failed, see NCCL output (NCCL_DEBUG=INFO) for details.");
  }
  // completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(nccl_ctx_.stream));
  entry.callback(Status::OK());
}

void NCCLController::Broadcast(TensorTableEntry& entry) {
  const int root_rank = entry.root_rank;
  // On root rank, ncclBcast sends data, on other ranks it receives data.
  int num_elements = entry.tensor->shape().num_elements();
  void* data_ptr;
  if (mpi_ctx_.rank_ == root_rank) {
    data_ptr = (void*)entry.tensor->data();
  } else {
    data_ptr = (void*)entry.output->data();
  }

  with_device device_guard(entry.device);
  int ret_code =
      ncclBcast(data_ptr, num_elements, GetNCCLDataType(entry.tensor),
                root_rank, nccl_ctx_.nccl_comm, nccl_ctx_.stream);
  if (ret_code != ncclSuccess) {
    throw std::runtime_error(
        "ncclBroadcast failed, see NCCL output (NCCL_DEBUG=INFO) for details.");
  }
  // completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(nccl_ctx_.stream));
  entry.callback(Status::OK());
}

#if HAVE_NCCL && NCCL_MINOR > 6
void NCCLController::NeighborAllgather(TensorTableEntry& entry) {
  // Note the order of recvcounts and displcments is by the oder of
  // mpi_ctx_.neighbor_in_ranks_ because of MPI_Neighbor_allgather and
  // 
  int* recvcounts = new int[mpi_ctx_.neighbor_indgree_];
  int* displcmnts = new int[mpi_ctx_.neighbor_indgree_];
  if (!mpi_ctx_.IsTopoSetup()) {
    throw std::runtime_error(
        "Topology has not been set yet cannot run neighbor_allgather");
  }
  mpi_ctx_.AllocateOutput(entry, recvcounts, Communicator::GRAPH);
  mpi_ctx_.SetDisplacements(recvcounts, displcmnts, Communicator::GRAPH);
  if (!CheckSameRecvSize(recvcounts, mpi_ctx_.neighbor_indgree_)) {
    throw std::runtime_error(
        "Neighbor_allgather/allreduce doesn't support varying lenght of vector. Please make "
        "sure the size of tensors is the same among all processes.");
  }

  const void* sendbuf = entry.tensor->data();
  int num_elements = entry.tensor->shape().num_elements();
  void* buffer_data = (void*)entry.output->data();

  Timeline* timeline_ptr;
  Status timeline_status = GetBluefogTimeline(timeline_ptr);

  // We need to explicitly set the device here.
  with_device device_guard(entry.device);

  timeline_ptr->ActivityStart(entry.tensor_name, "COMMUNICATE");

  // Pitfall: neighbor_allgather do not include itself.
  ncclGroupStart();
  for (int i = 0; i < mpi_ctx_.neighbor_indgree_; i++) {
    int recv_rank = mpi_ctx_.neighbor_in_ranks_[i];
    int recv_count = recvcounts[i];
    int target_disp = displcmnts[i];
    int element_size = mpi_ctx_.GetMPITypeSize(
        entry.tensor->dtype());  // Assume NCCL use same size as MPI
    void* recvbuf =
        (void*)(static_cast<char*>(buffer_data) + target_disp * element_size);
    NCCLCHECK(ncclRecv(recvbuf, recv_count, GetNCCLDataType(entry.tensor), recv_rank,
                       nccl_ctx_.nccl_comm, nccl_ctx_.stream));
  }
  for (int rank : mpi_ctx_.neighbor_out_ranks_) {
    NCCLCHECK(ncclSend(sendbuf, num_elements, GetNCCLDataType(entry.tensor),
                       rank, nccl_ctx_.nccl_comm, nccl_ctx_.stream));
  }
  ncclGroupEnd();

  // completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(nccl_ctx_.stream));

  delete[] recvcounts;
  delete[] displcmnts;
  timeline_ptr->ActivityEnd(entry.tensor_name);

  timeline_ptr->ActivityStart(entry.tensor_name, "CALLBACK");
  entry.callback(Status::OK());
  timeline_ptr->ActivityEnd(entry.tensor_name);
}

void NCCLController::NeighborAllreduce(TensorTableEntry& entry)  {
  // The communication pattern of neighbor_allreduce and neighbor_allgather are the same.
  // The difference happened at the callback phase.
  NeighborAllgather(entry);
}
#endif

}  // namespace common
}  // namespace bluefog
