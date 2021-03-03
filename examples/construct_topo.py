import bluefog.torch as bf
# from bluefog.common import topology_util
import networkx as nx
bf.init()
# dst_list = [i for i in range(bf.size()) if i != bf.rank()]

# dst_list = [(bf.rank()+1)%bf.size(), (bf.rank()+3)%bf.size()]

if bf.rank() == 0:
    dst_list = [1, 2]
elif bf.rank() == 1:
    dst_list = [0]
elif bf.rank() == 3:
    dst_list = [0]
else:
    dst_list = [0]

bf.set_topology(bf.topology_util.ExponentialTwoGraph(size=4))
print(f"{bf.rank()}: {dst_list}")
# print(bf.rank() in dst_list)
W = bf.infer_destination_source_ranks(
    src_list=dst_list, construct_adjacency_matrix=True)
G = nx.from_numpy_array(W, create_using=nx.DiGraph)
if bf.rank() == 0:
    print(W)
    print(bf.GetRecvWeights(G, bf.rank()))
