from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bluefog.common.util import check_extension

check_extension('bluefog.tensorflow', __file__, 'mpi_lib')

# pylint: disable = wrong-import-position
from bluefog.tensorflow.mpi_ops import init, shutdown
from bluefog.tensorflow.mpi_ops import size, local_size, rank, local_rank
from bluefog.tensorflow.mpi_ops import load_topology, set_topology
from bluefog.tensorflow.mpi_ops import mpi_threads_supported
from bluefog.tensorflow.mpi_ops import allreduce, broadcast, allgather
