import networkx as nx
from bluefog.common import topology_util
G = topology_util.StarGraph(16)
nx.draw_spring(G)
