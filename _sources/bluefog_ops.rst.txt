.. _Ops Explanation:

Bluefog Operations Explanation
==============================

The implementation of bluefog operations is built upon the MPI API. 
The naming of communication operations are all derived from MPI. However,
our usage and definition is slightly different from the MPI since our focus
is highly associated with the virtual topology of network.

The communication ops that bluefog supported can be catogorized into three types:

1. Collective Ops: ``broadcast``, ``allreduce``, ``allgather``.
2. Neighbor Colletive Ops: ``neighbor_allreduce``, ``neighbor_allgather``.
3. One-sided Communication Ops: ``win_create``, ``win_free``, ``win_sync``, ``win_put``, ``win_get``, ``win_accumulate``.

We use figure to illustrate all those ops with 
similar style as in `MPI tutorials blog`_. 
In the figure, we use *circle* to represent one process, which is exchangeablely called node,
agent, host, etc. under some circumstance, and use *square* to represent the data or tensor. 
The number inside of circle is the rank of that process and th number inside of square is the value of data.


Collective Ops
--------------
These three ``broadcast``, ``allreduce``, ``allgather`` ops are most basic collective MPI ops.
The bluefog implementation is almost exactly the same as the MPI definition. One small difference
is allreduce only support average and summation since we focused on the numerical calculation only.

allgather
#########

.. image:: _static/bf_allgather.png
    :alt: BluefogAllgatherExplanation
    :width: 350

allreduce
#########

.. image:: _static/bf_allreduce.png
    :alt: BluefogAllreduceExplanation
    :width: 300

broadcast
#########

.. image:: _static/bf_broadcast.png
    :alt: BluefogBroadcastExplanation
    :width: 450



Neighbor Colletive Ops
----------------------
Similar to their collective ops cousins, the behavior of neighbor collective ops is very similar,
except that their behavior is determined by the virtual topology as well. Loosenly speaking, 
allreduce and allgather are the same as running the neighbor_allreduce and neighbor_allgather 
over fully connected network. In the figure, we use the arrowed line to represent the connection of
virtual topology (notice it is the directed graph.)

neighbor_allgather
##################
.. image:: _static/bf_neighbor_allgather.png
    :alt: BluefogNeighborAllgatherExplanation
    :width: 600

neighbor_allreduce
##################
.. image:: _static/bf_neighbor_allreduce.png
    :alt: BluefogNeighborAllreduceExplanation
    :width: 600

.. Note::
   In the figure, we only show the neighbor_allreduce with average with uniform weight. Actually, our
   API allows for any weights for incoming edges. Check out API doc to see how to use it.


One-sided Communication Ops
---------------------------

To be added.

win_create
##########
.. image:: _static/bf_win_create.png
    :alt: BluefogWinCreateExplanation
    :width: 650

win_free
########
.. image:: _static/bf_win_free.png
    :alt: BluefogWinFreeExplanation
    :width: 650

win_put
#######
.. image:: _static/bf_win_put.png
    :alt: BluefogWinPutExplanation
    :width: 650

win_get
#######
.. image:: _static/bf_win_get.png
    :alt: BluefogWinGetExplanation
    :width: 650

win_accumulate
##############
.. image:: _static/bf_win_accum.png
    :alt: BluefogWinAccumExplanation
    :width: 650

win_sync
########
.. image:: _static/bf_win_sync.png
    :alt: BluefogWinSyncExplanation
    :width: 650

..  _MPI tutorials blog: https://mpitutorial.com/tutorials/
