#ifndef BLUEFOG_COMMON_OPERATIONS_H
#define  BLUEFOG_COMMON_OPERATIONS_H

#include <functional>
#include "common.h"

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

// C interface to return number of (in-) neighbor bluefog processes.
// Returns -1 if bluefog is not initialized or topology is not set.
int bluefog_neighbor_size();

// C interface to set the virtual topology for MPI graph communicator.
// Also, the corresponding graph communicator is created.
// Returns -1 if Bluefog is not initialized or failed.
int bluefog_set_topology(int indegree, const int* sources, 
                         int outdegree, const int* destinations);

// C interface to load the virtual topology for MPI graph communicator.
// Returns -1 if Bluefog is not initialized or failed.
int bluefog_load_topology(int* indegree, int*& sources, 
                          int* outdegree, int*& destinations);

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

Status EnqueuTensorWindowPut(std::shared_ptr<Tensor> tensor,
                             const std::string& name, 
                             const std::vector<int>& dst_ranks,
                             const int device,
                             StatusCallback callback);

Status EnqueuTensorWindowGet(std::shared_ptr<Tensor> tensor,
                             const std::string& name, 
                             const std::vector<int>& src_ranks, 
                             const int device,
                             StatusCallback callback);

Status WindowCreate(std::shared_ptr<Tensor> tensor,
                    std::vector<std::shared_ptr<Tensor>> neighbor_tensors,
                    const std::string& name, int device);

Status WindowSync(const std::string& name);

Status WindowFree(const std::string& name);

Status WindowFence(const std::string& name);

Status Barrier();

}  // namespace common
}  // namespace bluefog

#endif