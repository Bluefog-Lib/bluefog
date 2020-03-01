from typing import List, Tuple

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


def PowerTwoRingGraph(size: int) -> nx.DiGraph:
    """Each point only connected to a point such that the index difference is power of 2."""
    assert size > 0
    x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
    x /= x.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G

def MeshGrid2DGraph(size: int, shape: Tuple[int, int] = None) -> nx.DiGraph:
    """ 
    2D MeshGrid structure of a graph.
    Assume shape = (nrow, ncol)
    - When shape is provided, a meshgrid of nrow*ncol will be generated.
    - When shape is not provided, nrow and ncol will be the two closest factors of size.
      For example: size = 24, nrow and ncol will be 4 and 6, respectively.
      We assume  nrow will be equal to or smaller than ncol.
      If size is a prime number, nrow will be 1, and ncol will be size, which degrades the topology 
      into a linear one.
    """

    assert size > 0
    if shape is None:
        i = int(np.sqrt(size))
        while size%i != 0: i -= 1
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
    """Star structure of graph, i.e. all other ranks are connected to rank 0 (bidirection)."""
    assert size > 0
    topo = np.zeros((size, size))
    for i in range(size):
        topo[i, i] = 1 - 1 / size
        topo[center_rank, i] = 1 / size
        topo[i, center_rank] = 1 / size
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def RingGraph(size: int, left_connect: bool = False) -> nx.DiGraph:
    """Ring structure of graph (uniliteral)."""
    assert size > 0
    if size == 1:
        return nx.from_numpy_array(np.array([[1.0]]), create_using=nx.DiGraph)
    x = np.zeros(size)
    x[0] = 0.5
    if left_connect:
        x[-1] = 0.5
    else:
        x[1] = 0.5
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def BiRingGraph(size: int) -> nx.DiGraph:
    """Ring structure of graph (bidirection)."""
    assert size > 0
    if size == 1:
        return nx.from_numpy_array(np.array([[1.0]]), create_using=nx.DiGraph)
    if size == 2:
        return nx.from_numpy_array(np.array([[0.5,0.5],[0.5,0.5]]), create_using=nx.DiGraph)
    x = np.zeros(size)
    x[0] = 1/3.0
    x[-1] = 1/3.0
    x[1] = 1/3.0
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G

def GetWeights(topo: nx.DiGraph, rank: int) -> List[float]:
    """Return a List of (in-)weights of rank.
    Self weights is always put at the beginning. If there is no self-loop,
    the first element will be zero. The rest will be the same order as
    predecessors returned.
    """
    weight_matrix = nx.to_numpy_array(topo)
    ret_weights = [0.0]
    for src_rank in topo.predecessors(rank):
        if src_rank == rank:
            ret_weights[0] = weight_matrix[src_rank, rank]
        else:
            ret_weights.append(weight_matrix[src_rank, rank])
    return ret_weights
