import networkx as nx
from bluefog.common import topology_util
G = topology_util.RingGraph(16)
nx.draw_circular(G)
