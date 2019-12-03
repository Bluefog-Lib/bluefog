from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

def mpi_env_rank_and_size():
    """Get MPI rank and size from environment variables and return them as a
    tuple of integers.

    Most MPI implementations have an `mpirun` or `mpiexec` command that will
    run an MPI executable and set up all communication necessary between the
    different processors. As part of that set up, they will set environment
    variables that contain the rank and size of the MPI_COMM_WORLD
    communicator.
    """
    rank_env = "PMI_RANK OMPI_COMM_WORLD_RANK".split()
    size_env = "PMI_SIZE OMPI_COMM_WORLD_SIZE".split()

    for rank_var, size_var in zip(rank_env, size_env):
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)

    # Default to rank zero and size one if there are no environment variables
    return 0, 1
