import atexit
import networkx

from bluefog.common.operations import InitializeBlueFogOnce
from bluefog.common.common import BlueFogError


class BlueFogBasics(object):
    """Wrapper class for the basic BlueFog API."""

    def __init__(self, *args):
        del args
        self.bf_global_state = None

    def init(self, topology: networkx.DiGraph = None, comm=None) -> None:
        """A function that initializes BlueFog.

        Args:
          Topo: A networkx. DiGraph object to decide the topology. If not provided
            a default power_two_ring structure is used.
          comm: List specifying ranks for the communicator, relative to the MPI_COMM_WORLD
            communicator OR the MPI communicator to use. Given communicator will be duplicated.
            If None, BlueFog will use MPI_COMM_WORLD Communicator.
        """
        del comm
        self.bf_global_state = InitializeBlueFogOnce(self.bf_global_state, topology)
        atexit.register(self.shutdown)

    def shutdown(self) -> int:
        """A function that shuts BlueFog down."""
        # TODO(ybc) Ensure the MPI will be init only once and run Finialize without error at exit().
        # self.bf_global_state.context.shutdown()
        self.bf_global_state.initialized = False
        self.bf_global_state.shut_down = True

    def size(self) -> int:
        """A function that returns the number of BlueFog processes.

        Returns:
          An integer scalar containing the number of BlueFog processes.
        """
        if not self.bf_global_state.initialized:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return self.bf_global_state.controller.size

    def local_size(self) -> int:
        """A function that returns the number of BlueFog processes within the
        node the current process is running on.

        Returns:
          An integer scalar containing the number of local BlueFog processes.
        """
        if not self.bf_global_state.initialized:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return self.bf_global_state.controller.local_size

    def rank(self) -> int:
        """A function that returns the BlueFog rank of the calling process.

        Returns:
          An integer scalar with the BlueFog rank of the calling process.
        """
        if not self.bf_global_state.initialized:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return self.bf_global_state.controller.rank

    def local_rank(self) -> int:
        """A function that returns the local BlueFog rank of the calling process, within the
        node that it is running on. For example, if there are seven processes running
        on a node, their local ranks will be zero through six, inclusive.

        Returns:
          An integer scalar with the local BlueFog rank of the calling process.
        """
        if not self.bf_global_state.initialized:
            raise ValueError("BlueFog has not been initialized; use bf.init().")
        return self.bf_global_state.controller.local_rank

    def mpi_threads_supported(self) -> bool:
        """A function that returns a flag indicating whether MPI multi-threading is supported.

        If MPI multi-threading is supported, users may mix and match BlueFog usage with other
        MPI libraries, such as `mpi4py`.

        Returns:
          A boolean value indicating whether MPI multi-threading is supported.
        """
        if not self.bf_global_state.initialized:
            raise ValueError(
                "BlueFog has not been initialized; use bf.init().")
        return self.bf_global_state.controller.mpi_threads_supported

    def load_topology(self) -> networkx.DiGraph:
        """A funnction that return the virtual topology MPI used.

        Raises:
            BlueFogError: If Bluefog has not been initialized.

        Returns:
            networkx.DiGraph: Topology
        """
        if not self.bf_global_state.initialized:
            raise BlueFogError(
                "BlueFog has not been initialized; use bf.init().")
        return self.bf_global_state.topology

    def set_topology(self, topology: networkx.DiGraph):
        """A funnction that set the virtual topology MPI used.

        Raises:
            BlueFogError: If Bluefog has not been initialized.

        Returns:
            networkx.DiGraph: Topology
        """
        if not self.bf_global_state.initialized:
            raise BlueFogError(
                "BlueFog has not been initialized; use bf.init().")
        if not isinstance(topology, networkx.DiGraph):
            raise ValueError("Topology has to be a networkx.DiGraph object.")

        self.bf_global_state.context.SetDistGraphComm(topology)
        self.bf_global_state.topology = topology
