Bluefog Environment Variables
=============================

Bluefog has several environment variables to tune or enable some functionality for specific usage.


Debug
-----

If you want to see more verbose logs of Bluefog, you can set

.. code-block:: bash

    export BLUEFOG_LOG_LEVEL=info

There are six different levels -- `trace`, `debug`, `info`, `warning`, `error`, and `fatal`. Default is `warning`.

You can also hide the time in logs by setting:

.. code-block:: bash

    export BLUEFOG_LOG_HIDE_TIME=1



Performance
-----------

Our implementation is based upon the architecture of Horovod. Hence, we also have the tensor fusion functionality, which
batches small allreduce/neighbor_allreduce operations into one to improve the performance. Cycle time is anthor tuning parameter 
to determine within the certain period of time, the ready tensor will be fused into one if it doesnot exceed the threshold of fusion.

* BLUEFOG_FUSION_THRESHOLD
* BLUEFOG_CYCLE_TIME

The fusion threshold is based on the Byte size and cycle time is based on the milliseconds.

**Timeline**:

You can set `BLUEFOG_TIMELINE` with some filename to turn on the timeline. See our timeline document for more details.

.. code-block:: bash
    
    export BLUEFOG_TIMELINE=/path/filename


**MPI Thread Support**:

By default, we will ask for MPI_THREAD_SERIALIZED -- The process may be 
multi-threaded, but only the main thread will make MPI calls (all MPI calls 
are funneled to the main thread). The main reason we use this level is MPI win 
ops doesn't have good support for multiple processes on multiple hosts unless 
running under a system with an RDMA  capable network such as Infiniband.

.. code-block:: bash

   export  BLUEFOG_MPI_THREAD_LEVEL=2

The environment variable BLUEFOG_MPI_THREAD_LEVEL has to be one of the values 0, 1, 2, or 3 
-- corresponding to `MPI_THREAD_SINGLE`, `MPI_THREAD_FUNNELED`, `MPI_THREAD_SERIALIZED`, or `MPI_THREAD_MULTIPLE`

**Ops Running Backend**:

If you build the Bluefog with NCCL, most communication operations will be executed through the NCCL. However, you still can force
it to be run by MPI through set following environment variable(s) to be 1:

* BLUEFOG_ALLREDUCE_BY_MPI
* BLUEFOG_ALLGATHER_BY_MPI
* BLUEFOG_BROADCAST_BY_MPI
* BLUEFOG_NEIGHBOR_ALLREDUCE_BY_MPI
* BLUEFOG_NEIGHBOR_ALLGATHER_BY_MPI

GPU tensor will send through GPU directly if it is supported. If you want to specify it to communication through the CPU, set
following environment variable to be 1 or 0:

* BLUEFOG_OPS_ON_CPU
* BLUEFOG_WIN_ON_GPU

**Misc**:

Due to unclear reason that mpi_put/get/accumlate under the
mpi_lock epoch cannot send too long vector in one time, we
define this number as the maximum size of win_ops can send.

* BLUEFOG_MAX_WIN_SENT_LENGTH (Default: 2000)

When the NCCL implementation is used, the callback functions are executed through
a thread pool. The size of thread pool can be controlled by following:

* BLUEFOG_NUM_FINALIZER_THREADS (Default: 1)

Installation
------------

**MPI Related**:

* BLUEFOG_MPICXX_SHOW -- Specify the location of MPI include and library location (Default: "mpicxx -show").

**CUDA Related**:

If the cuda is detected, such as pytorch supports CUDA, Bluefog will be built with CUDA automatically. You don't need
to specify any of followings unless the installation location of CUDA is not at standard position.

* BLUEFOG_CUDA_HOME -- Specify the CUDA Home location, i.e. the parent folder of include and library.
* BLUEFOG_CUDA_INCLUDE -- Specify the CUDA include location.
* BLUEFOG_CUDA_LIB -- Specify the CUDA library location.

**NCCL Related**:

Unlike CUDA, you have to manually set `BLUEFOG_WITH_NCCL=1` during the installation if you want to build Bluefog with
NCCL support. The other locaiton variables is similar to CUDA ones.

* BLUEFOG_WITH_NCCL -- Set 1 to let bluefog built with NCCL (Default: 0),
* BLUEFOG_NCCL_HOME -- Specify the NCCL Home location, i.e. the parent folder of include and library.
* BLUEFOG_NCCL_INCLUDE -- Specify the NCCL include location.
* BLUEFOG_NCCL_LIB -- Specify the NCCL library location.
