.. Bluefog documentation master file, created by
   sphinx-quickstart on Mon Dec  2 20:20:32 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bluefog
=======

.. image:: _static/bf_logo_h.png
   :width: 450
   :align: center
   :alt: Bluefog Logo

OVERVIEW
--------

BlueFog is a high-performance distributed training framework for PyTorch built with **decentralized optimization** algorithms. 
The goal of Bluefog is to make decentralized algorithms easy to use, fault-tolerant, friendly to heterogeneous environment, 
and even faster than training frameworks built with parameter server, or ring-allreduce. 

In each communication stage, neither the typical star-shaped parameter-server toplogy, nor the pipelined ring-allreduce topology is used, which
is fundamentally different from other popular distributed training frameworks, such as DistributedDataParallel provided by PyTorch, Horovod, BytePS, etc. 
Instead, BlueFog will exploit a virtual and probably dynamic network topology (that can be in any shape) to achieve most communication efficiency.

..
    
    Main Idea: Replace expensive allreduce averaging over gradients by cheap neighbor averaging over parameters

For each training iteration, one process (or agent) will update its model with information received from its **direct** neighbors defined by the virtual topology. It is observed all communications only occur over the predefied virtual topolgy and no global communication is required. This is why the algorithms is named *decentralized*. 
Decentralized training algorithms are proved in literature that it can converge to the same solution as their standard centralized counterparts. 

The topology decides the communication efficiency. BlueFog supports both **static** topology and **dynamic** topology usages. After tremendous trials, the dynamic Exponential-2 graph is observed to achieve the best performance
if the number of agents is the power of 2, such as 4, 32, 128 agents. In Exponential-2 graph, each agent will 
communicates with the neighbors that are  2 :sup:`0`, 2 :sup:`1`, ..., 2 :sup:`t` hops away. **Dynamic** toplogy means all agents select
one neighbor only in one iteration and select next neighbor in next iteration as illustrated in the following figure:

.. raw:: html

    <p align="center"><img src="https://user-images.githubusercontent.com/16711681/97928035-04654400-1d1b-11eb-91d2-2da890b4522e.png" alt="one-peer-exp2" width="650"/></p>

In this scenario, the communcation cost for each iteration is only one unit delay, one standard parameter size to transmit and no communication conflict happens, which is better than what parameter server or ring-allreduce promises. As for loss and accuracy guarantees, please check out our theoratical paper. [Will add a full tutorial soon].

.. toctree::
   :maxdepth: 1
   :caption: INSTALLATION

   Installing Bluefog <install>


.. toctree::
   :maxdepth: 2
   :caption: API

   Bluefog Torch API <torch_api>
   Bluefog Topology API <topo_api>

.. toctree::
   :maxdepth: 1
   :caption: More Information

   Bluefog Ops Explanation <bluefog_ops>
   Bluefog Performance <performance>
   Static and Dynamic Topology Neighbor Averaging <neighbor_average>
   Lauching Application Through bfrun <running>
   Bluefog Docker Usage <docker>
   Bluefog Environment Variable <env_variable>
   Bluefog Timeline <timeline>
   Spectrum of Machine Learning Algorithm<alg_spectrum>
   Codebase Structure <code_structure>
   Development Guide <devel_guide>

.. _openmpi: https://www.open-mpi.org/
.. _examples: https://github.com/Bluefog-Lib/bluefog/tree/master/examples
.. _AWS: https://aws.amazon.com/about-aws/whats-new/2018/12/introducing-amazon-ec2-p3dn-instances-our-most-powerful-gpu-instance-yet/

