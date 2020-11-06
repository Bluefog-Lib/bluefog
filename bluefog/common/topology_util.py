# Copyright 2020 Bluefog Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import List, Tuple, Dict, Iterator, Optional

import math
import numpy as np
import networkx as nx


def IsTopologyEquivalent(topo1: nx.DiGraph, topo2: nx.DiGraph) -> bool:
    """ Determine two topologies are equivalent or not.

    Notice we do not check two topologies are isomorphism. Instead checking
    the adjacenty matrix is the same only.
    """
    if topo1 is None or topo2 is None:
        return False
    if topo1.number_of_nodes() != topo2.number_of_nodes():
        return False
    if topo1.number_of_edges() != topo2.number_of_edges():
        return False
    A1 = nx.to_numpy_matrix(topo1).ravel()
    A2 = nx.to_numpy_matrix(topo2).ravel()
    return (A1 == A2).all()


def GetRecvWeights(topo: nx.DiGraph, rank: int) -> Tuple[float, Dict[int, float]]:
    """Return a Tuple of self_weight and neighbor_weights for receiving dictionary."""
    weight_matrix = nx.to_numpy_array(topo)
    self_weight = 0.0
    neighbor_weights = {}
    for src_rank in topo.predecessors(rank):
        if src_rank == rank:
            self_weight = weight_matrix[src_rank, rank]
        else:
            neighbor_weights[src_rank] = weight_matrix[src_rank, rank]
    return self_weight, neighbor_weights


def GetSendWeights(topo: nx.DiGraph, rank: int) -> Tuple[float, Dict[int, float]]:
    """Return a Tuple of self_weight and neighbor_weights for sending dictionary."""
    weight_matrix = nx.to_numpy_array(topo)
    self_weight = 0.0
    neighbor_weights = {}
    for recv_rank in topo.successors(rank):
        if recv_rank == rank:
            self_weight = weight_matrix[rank, recv_rank]
        else:
            neighbor_weights[recv_rank] = weight_matrix[rank, recv_rank]
    return self_weight, neighbor_weights


def ExponentialTwoGraph(size: int) -> nx.DiGraph:
    """Generate graph topology such that each points only
    connected to a point such that the index difference is the power of 2.

    Example: A ExponentialTwoGraph with 12 nodes:

    .. plot::
        :context: close-figs

        >>> import networkx as nx
        >>> from bluefog.common import topology_util
        >>> G = topology_util.ExponentialTwoGraph(12)
        >>> nx.draw_circular(G)
    """
    assert size > 0
    x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
    x /= x.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def isPowerOf(x, base):
    assert isinstance(base, int), "Base has to be a integer."
    assert base > 1, "Base has to a interger larger than 1."
    assert x > 0
    if (base ** int(math.log(x, base))) == x:
        return True
    return False


def ExponentialGraph(size: int, base: int = 2) -> nx.DiGraph:
    """Generate graph topology such that each points only
    connected to a point such that the index difference is power of base. (Default is 2)

    Example: A ExponentialGraph with 12 nodes:

    .. plot::
        :context: close-figs

        >>> import networkx as nx
        >>> from bluefog.common import topology_util
        >>> G = topology_util.ExponentialGraph(12)
        >>> nx.draw_circular(G)
    """
    x = [1.0]
    for i in range(1, size):
        if isPowerOf(i, base):
            x.append(1.0)
        else:
            x.append(0.0)
    x = np.array(x)
    x /= x.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def SymmetricExponentialGraph(size: int, base: int = 4) -> nx.DiGraph:
    """
     Generate symmeteric graph topology such that for the first half of nodes
     only connected to a point such that the index difference is power of base (Default is 4)
     and the connectivity for the second half of nodes just mirrored to the first half.

    Example: A SymmetricExponentialGraph with 12 nodes

    .. plot::
        :context: close-figs

        >>> import networkx as nx
        >>> from bluefog.common import topology_util
        >>> G = topology_util.SymmetricExponentialGraph(12)
        >>> nx.draw_circular(G)
    """
    x = [1.0]
    for i in range(1, size):
        index = i if i <= size // 2 else size - i
        if isPowerOf(index, base):
            x.append(1.0)
        else:
            x.append(0.0)
    x = np.array(x)
    x /= x.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def MeshGrid2DGraph(size: int, shape: Optional[Tuple[int, int]] = None) -> nx.DiGraph:
    """Generate 2D MeshGrid structure of graph.

    Assume shape = (nrow, ncol), when shape is provided, a meshgrid of nrow*ncol will be generated.
    when shape is not provided, nrow and ncol will be the two closest factors of size.

    For example: size = 24, nrow and ncol will be 4 and 6, respectively.
    We assume  nrow will be equal to or smaller than ncol.
    If size is a prime number, nrow will be 1, and ncol will be size, which degrades the topology
    into a linear one.

    Example: A MeshGrid2DGraph with 16 nodes:

    .. plot::
        :context: close-figs

        >>> import networkx as nx
        >>> from bluefog.common import topology_util
        >>> G = topology_util.MeshGrid2DGraph(16)
        >>> nx.draw_spring(G)
    """

    assert size > 0
    if shape is None:
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size//i)
    nrow, ncol = shape
    assert size == nrow*ncol, "The shape doesn't match the size provided."
    topo = np.zeros((size, size))
    for i in range(size):
        topo[i][i] = 1.0
        if (i+1) % ncol != 0:
            topo[i][i+1] = 1.0
            topo[i+1][i] = 1.0
        if i+ncol < size:
            topo[i][i+ncol] = 1.0
            topo[i+ncol][i] = 1.0

    # According to Hasting rule (Policy 1) in https://arxiv.org/pdf/1702.05122.pdf
    # The neighbor definition in the paper is different from our implementation,
    # which includes the self node.
    topo_neighbor_with_self = [np.nonzero(topo[i])[0] for i in range(size)]
    for i in range(size):
        for j in topo_neighbor_with_self[i]:
            if i != j:
                topo[i][j] = 1.0/max(len(topo_neighbor_with_self[i]),
                                     len(topo_neighbor_with_self[j]))
        topo[i][i] = 2.0-topo[i].sum()
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def StarGraph(size: int, center_rank: int = 0) -> nx.DiGraph:
    """Generate star structure of graph.

    All other ranks are connected to the center_rank. The connection is
    bidirection, i.e. if the weight from node i to node j is non-zero, so
    is the weight from node j to node i.

    Example: A StarGraph with 16 nodes:

    .. plot::

        >>> import networkx as nx
        >>> from bluefog.common import topology_util
        >>> G = topology_util.StarGraph(16)
        >>> nx.draw_spring(G)
    """
    assert size > 0
    topo = np.zeros((size, size))
    for i in range(size):
        topo[i, i] = 1 - 1 / size
        topo[center_rank, i] = 1 / size
        topo[i, center_rank] = 1 / size
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def RingGraph(size: int, connect_style: int = 0) -> nx.DiGraph:
    """Generate ring structure of graph (uniliteral).
    Argument connect_style should be an integer between 0 and 2, where
    0 represents the bi-connection, 1 represents the left-connection,
    and 2 represents the right-connection.

    Example: A RingGraph with 16 nodes:

    .. plot::

        >>> import networkx as nx
        >>> from bluefog.common import topology_util
        >>> G = topology_util.RingGraph(16)
        >>> nx.draw_circular(G)
    """
    assert size > 0
    assert connect_style >= 0 and connect_style <= 2, \
        "connect_style has to be int between 0 and 2, where 1 " \
        "for bi-connection, 1 for left connection, 2 for right connection."
    if size == 1:
        return nx.from_numpy_array(np.array([[1.0]]), create_using=nx.DiGraph)
    if size == 2:
        return nx.from_numpy_array(np.array([[0.5, 0.5], [0.5, 0.5]]), create_using=nx.DiGraph)

    x = np.zeros(size)
    x[0] = 0.5
    if connect_style == 0:  # bi-connection
        x[0] = 1/3.0
        x[-1] = 1/3.0
        x[1] = 1/3.0
    elif connect_style == 1:  # left-connection
        x[-1] = 0.5
    elif connect_style == 2:  # right-connection
        x[1] = 0.5
    else:
        raise ValueError("Connect_style has to be int between 0 and 2")

    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def FullyConnectedGraph(size: int) -> nx.DiGraph:
    """Generate fully connected structure of graph.
    For example, a FullyConnectedGraph with 16 nodes:

    Example: A FullyConnectedGraph 16 nodes:

    .. plot::

        >>> import networkx as nx
        >>> from bluefog.common import topology_util
        >>> G = topology_util.FullyConnectedGraph(16)
        >>> nx.draw_spring(G)
    """
    assert size > 0
    x = np.array([1/size] * size)
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def IsRegularGraph(topo: nx.DiGraph) -> bool:
    """Dtermine a graph is regular or not, i.e. all nodes have the same degree."""
    degree = topo.degree(0)
    for rank in range(1, topo.number_of_nodes()):
        if topo.degree(rank) != degree:
            return False
    return True


def GetDynamicSendRecvRanks(
        topo: nx.DiGraph, self_rank: int) -> Iterator[Tuple[List[int], List[int]]]:
    """A utility function to generate 1-outoging send rank and corresponding recieving rank(s).

    Args:
        topo (nx.DiGraph): The base topology to generate dynamic send and receive ranks.
        self_rank (int): The self rank.

    Yields:
        Iterator[Tuple[List[int], List[int]]]: send_ranks, recv_ranks.

    Example:

        >>> from bluefog.common import topology_util
        >>> topo = topology_util.PowerTwoRingGraph(10)
        >>> gen = topology_util.GetDynamicSendRecvRanks(topo, 0)
        >>> for _ in range(10):
        >>>     print(next(gen))
    """
    # Generate all outgoing ranks sorted by clock-wise. (Imagine all ranks put on a clock.)
    size = topo.number_of_nodes()
    sorted_send_ranks = []
    for rank in range(size):
        sorted_ranks = sorted(topo.successors(rank),
                              key=lambda r, rk=rank: r-rk if r >= rk else r-rk+size)
        if sorted_ranks[0] == rank:
            sorted_ranks = sorted_ranks[1:]  # remove the self-loop
        sorted_send_ranks.append(sorted_ranks)

    self_degree = topo.out_degree(self_rank) - 1
    index = 0
    while True:
        send_rank = sorted_send_ranks[self_rank][index % self_degree]
        recv_ranks = []
        for other_rank in range(size):
            if other_rank == self_rank:
                continue
            degree = topo.out_degree(other_rank) - 1
            if sorted_send_ranks[other_rank][index % degree] == self_rank:
                recv_ranks.append(other_rank)

        yield [send_rank], recv_ranks
        index += 1


def GetExp2DynamicSendRecvMachineRanks(
        world_size: int, local_size: int, self_rank: int, local_rank: int
    ) -> Iterator[Tuple[List[int], List[int]]]:
    """
    A utility function to generate 1-outgoing send machine id and corresponding recieving
    machine id(s) for Exponentia-2 topology.

    Args:
        world_size (int): the size of all nodes; world_size = num_machines * nodes_per_machine
        local_size (int): number of nodes in each machine
        self_rank (int): The self rank.
        local_rank (int): The self local rank.

    Yields:
        Iterator[Tuple[List[int], List[int]]]: send_machine_ids, recv_machine_ids.

    Warning:
        This function should be used under homogeneous enviroment only, i.e. all machines have
        the same number of local processes.
    """
    assert (self_rank % local_size) == local_rank, \
        "It should be used under homogeneous environment only."
    assert (world_size % local_size) == 0, \
        "It should be used under homogeneous environment only."
    assert world_size > local_size, \
        "It should be used under at least two machines case."

    machine_id = self_rank // local_size
    machine_size = world_size // local_size
    exp_2_size = int(np.log2(machine_size-1)) if machine_size > 1 else 0
    index = 0
    while True:
        machine_dist = 2**(index % (exp_2_size + 1))
        send_machine_rank = (machine_id + machine_dist) % machine_size
        recv_machine_ranks = (machine_id - machine_dist) % machine_size
        yield [send_machine_rank], [recv_machine_ranks]
        index += 1


def GetInnerOuterRingDynamicSendRecvRanks(
        world_size: int, local_size: int, self_rank: int
    ) -> Iterator[Tuple[List[int], List[int]]]:
    """
    A utility function to generate 1-outgoing send rank and corresponding recieving rank(s)
    for Inner-Ring-Outer-Ring topology.

    Args:
        world_size (int): the size of all nodes; world_size = num_machines * nodes_per_machine
        local_size (int): number of nodes in each machine
        self_rank (int): The self rank.

    Yields:
        Iterator[Tuple[List[int], List[int]]]: send_ranks, recv_ranks.

    Example:

        >>> from bluefog.common import topology_util
        >>> world_size, local_size = bf.size(), bf.local_size()
        >>> gen = topology_util.GetInnerOuterRingDynamicSendRecvRanks(world_size, local_size, 0)
        >>> for _ in range(10):
        >>>     print(next(gen))
    """
    num_machines = world_size//local_size
    nodes_per_machine = local_size
    assert world_size % local_size == 0, "It should be used under homogeneous environment only."
    assert local_size > 2, "Do no support the case where nodes_per_machine is equal or " \
        "less than 2. Consider use hierarchical_neighbor_allreduce or GetDynamicSendRecvRanks."

    index = 0
    while True:
        machine_id = self_rank // nodes_per_machine
        local_rank_id = self_rank % nodes_per_machine
        local_rank_to_go_outside_id = index % nodes_per_machine

        if local_rank_to_go_outside_id == local_rank_id:
            # find send_rank
            target_machine_id = (machine_id + 1) % num_machines
            target_rank_id = target_machine_id * nodes_per_machine + local_rank_id
            send_rank = target_rank_id

            # find recv_rank
            source_machine_id = (machine_id - 1) % num_machines
            source_rank_id = source_machine_id * nodes_per_machine + local_rank_id
            recv_rank = source_rank_id

        else:
            # find send_rank
            target_local_rank_id = (local_rank_id + 1) % nodes_per_machine
            if target_local_rank_id == local_rank_to_go_outside_id:
                target_local_rank_id = (
                    target_local_rank_id + 1) % nodes_per_machine
            target_rank_id = target_local_rank_id + machine_id * nodes_per_machine
            send_rank = target_rank_id

            # find recv_rank
            source_local_rank_id = (local_rank_id - 1) % nodes_per_machine
            if source_local_rank_id == local_rank_to_go_outside_id:
                source_local_rank_id = (
                    source_local_rank_id - 1) % nodes_per_machine
            source_rank_id = source_local_rank_id + machine_id * nodes_per_machine
            recv_rank = source_rank_id

        yield [send_rank], [recv_rank]
        index += 1


def GetInnerOuterExpo2DynamicSendRecvRanks(
        world_size: int, local_size: int, self_rank: int
    ) -> Iterator[Tuple[List[int], List[int]]]:
    """
    A utility function to generate 1-outgoing send rank and corresponding recieving rank(s)
    for Inner-Exp2-Outer-Exp2 ring topology.

    Args:
        world_size (int): the size of all nodes; world_size = num_machines * nodes_per_machine
        local_size (int): number of nodes in each machine
        self_rank (int): The self rank.

    Yields:
        Iterator[Tuple[List[int], List[int]]]: send_ranks, recv_ranks.

    Example:

        >>> from bluefog.common import topology_util
        >>> world_size, local_size = bf.size(), bf.local_size()
        >>> gen = topology_util.GetInnerOuterExpo2DynamicSendRecvRanks(world_size, local_size, 0)
        >>> for _ in range(10):
        >>>     print(next(gen))
    """
    num_machines = world_size//local_size
    nodes_per_machine = local_size
    assert world_size % local_size == 0, "It should be used under homogeneous environment only."
    assert local_size > 2, "Do no support the case where nodes_per_machine is equal or " \
        "less than 2. Consider use hierarchical_neighbor_allreduce or GetDynamicSendRecvRanks."

    exp_2_out_size = int(np.log2(num_machines-1))
    if nodes_per_machine == 2:
        exp_2_in_size = 0
    else:
        # -2 because we need to remove outgoing node
        exp_2_in_size = int(np.log2(nodes_per_machine-2))  

    index = 0
    while True:
        machine_id = self_rank // nodes_per_machine
        local_rank_id = self_rank % nodes_per_machine
        local_rank_to_go_outside_id = index % nodes_per_machine

        if local_rank_to_go_outside_id == local_rank_id:
            # Note: currently design is still not very good. Because some local rank i may NEVER
            # directly talk to other machine's local rank i. Example:
            #
            # Assume num_machines=16, nodes_per_machine=4, and self_rank=1, then we know that
            # exp_2_out_size=3, and local_rank_id=1. If this branch is reached,
            # local_rank_to_go_outside_id=1, and index % (exp_2_out_size+1)=1, resulting in
            # next_machine_dist always equal to 2.
            next_machine_dist = 2**(index % (exp_2_out_size+1))
            # find send_rank
            target_machine_id = (machine_id + next_machine_dist) % num_machines
            target_rank_id = target_machine_id * nodes_per_machine + local_rank_id
            send_rank = target_rank_id

            # find recv_rank
            source_machine_id = (machine_id - next_machine_dist) % num_machines
            source_rank_id = source_machine_id * nodes_per_machine + local_rank_id
            recv_rank = source_rank_id

        else:
            # Distance from self to out-rank:
            dist_to_out = (local_rank_to_go_outside_id -
                            local_rank_id) % nodes_per_machine
            next_inner_dist = 2**(index % (exp_2_in_size + 1))
            if next_inner_dist >= dist_to_out:
                next_inner_dist += 1

            # find send_rank
            target_local_rank_id = (local_rank_id +
                                    next_inner_dist) % nodes_per_machine
            target_rank_id = target_local_rank_id + machine_id * nodes_per_machine
            send_rank = target_rank_id

            reverse_inner_dist = 2**(index % (exp_2_in_size + 1))
            reverse_dist_to_out = (
                local_rank_id - local_rank_to_go_outside_id) % nodes_per_machine
            if reverse_inner_dist >= reverse_dist_to_out:
                reverse_inner_dist += 1

            # find recv_rank
            source_local_rank_id = (local_rank_id -
                                    reverse_inner_dist) % nodes_per_machine
            source_rank_id = source_local_rank_id + machine_id * nodes_per_machine
            recv_rank = source_rank_id

        yield [send_rank], [recv_rank]
        index += 1
