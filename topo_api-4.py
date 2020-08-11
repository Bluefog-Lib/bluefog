import networkx as nx
from bluefog.common import topology_util
G = topology_util.MeshGrid2DGraph(16)
nx.draw_spring(G)
