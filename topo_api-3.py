import networkx as nx
from bluefog.common import topology_util
G = topology_util.SymmetricPowerGraph(12)
nx.draw_circular(G)
