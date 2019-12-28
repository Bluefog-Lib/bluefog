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
    x = [1 if i & (i - 1) == 0 else 0 for i in range(size)]
    x[0] = 0
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def StartGraph(size: int) -> nx.DiGraph:
    """Star structure of graph, i.e. all other ranks are connected to rank 0 (bidirection)."""
    return nx.star_graph(size, create_using=nx.Graph).to_directed()


def RingGraph(size: int) -> nx.DiGraph:
    """Ring structure of graph (uniliteral)."""
    return nx.cycle_graph(size, create_using=nx.DiGraph)


def BiRingGraph(size: int) -> nx.DiGraph:
    """Ring structure of graph (bidirection)."""
    return nx.cycle_graph(size, create_using=nx.Graph).to_directed()
