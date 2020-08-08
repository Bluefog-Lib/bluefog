import networkx as nx
from bluefog.common import topology_util
G = topology_util.FullyConnectedGraph(16)
nx.draw_spring(G)
