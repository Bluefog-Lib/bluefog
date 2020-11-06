Topology Related Utility Functions
==================================

We provide several popular static and dynamic topologies.

* Static Topology
    * ExponentialGraph, ExponentialTwoGraph
    * SymmetricExponentialGraph
    * MeshGrid2DGraph
    * StarGraph, RingGraph
    * FullyConnectedGraph

* Dynamic Topology
    * GetDynamicSendRecvRanks
    * GetExp2DynamicSendRecvMachineRanks
    * GetInnerOuterRingDynamicSendRecvRanks
    * GetInnerOuterExpo2DynamicSendRecvRanks

* Utility Function
    * IsRegularGraph
    * IsTopologyEquivalent
    * GetSendWeights, GetRecvWeights

You can also write your own topology strategy as long as your static topology function returns
a `networkx.DiGraph <https://networkx.org/documentation/stable/reference/classes/digraph.html>`_ 
object and dynamic topology function (generator more accurately) yields
a list of send neighbor and receive neighbor ranks in each call.

.. automodule:: bluefog.common.topology_util
    :exclude-members: List, Tuple, Dict, Iterator, isPowerOf