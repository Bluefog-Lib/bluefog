Torch Module (API Reference)
============================

All APIs can be roughly categorized into 5 classes:

* Bluefog Basic Operations:
    * init, shutdown, 
    * size, local_size, rank, local_rank, is_homogeneous
    * load_topology, set_topology, in_neighbor_ranks, out_neighbor_ranks
* High-level Optimizer Wrappers: 
    * DistributedGradientAllreduceOptimizer
    * DistributedAllreduceOptimizer
    * DistributedNeighborAllreduceOptimizer
    * DistributedHierarchicalNeighborAllreduceOptimizer
    * DistributedWinPutOptimizer
* Low-level Synchronous Communication Operations:
    * allreduce, allreduce_nonblocking, allreduce\_, allreduce_nonblocking\_
    * allgather, allgather_nonblocking
    * broadcast, broadcast_nonblocking, broadcast\_, broadcast_nonblocking\_
    * neighbor_allgather, neighbor_allgather_nonblocking
    * neighbor_allreduce, neighbor_allreduce_nonblocking
    * hierarchical_neighbor_allreduce, hierarchical_neighbor_allreduce_nonblocking
    * poll, synchronize, barrier
* Low-level Asynchronous Communication Operations:
    * win_create, win_free, win_update, win_update_then_collect
    * win_put_nonblocking, win_put
    * win_get_nonblocking, win_get
    * win_accumulate_nonblocking, win_accumulate
    * win_wait, win_poll, win_mutex
* Other miscellaneous and utility functions:
    * broadcast_optimizer_state, broadcast_parameters, allreduce_parameters
    * timeline_start_activity, timeline_end_activity
    * nccl_built, mpi_threads_supported, unified_mpi_window_model_supported

.. automodule:: bluefog.torch
    :members:
    :exclude-members: check_extension
    :member-order: bysource