from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bluefog.common.basics_c import BlueFogBasics

_basics = BlueFogBasics(__file__, 'mpi_lib')

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank
mpi_threads_supported = _basics.mpi_threads_supported
load_topology = _basics.load_topology
set_topology = _basics.set_topology
