BlueFog
=======

.. image:: https://travis-ci.com/Bluefog-Lib/bluefog.svg?branch=master
    :target: https://travis-ci.com/Bluefog-Lib/bluefog

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :alt: License

.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    
.. raw:: html

    <p align="center"><img src="https://user-images.githubusercontent.com/65107588/82258821-62d66b80-990f-11ea-9393-bf5456af67e6.png" alt="Logo" width="450"/></p>
    
BlueFog is a high-performance distributed training framework built with **decentralized optimization** algorithms. The goal of Bluefog is to make decentralized algorithms easy to use, fault-tolerant, friendly to heterogeneous environment, and even faster than training frameworks built with parameter server, or ring-allreduce.

Performance
-----------

Below are the charts representing the performance of BlueFog that was done on ResNet50 benchmark. Each machine has 8 V100 GPUs (64GB memory) with NVLink-enabled and the inter-connected communication speed is 25Gbps. This is the same hardware setup you can get on AWS_. We test the scaling efficiency with a batch size of 64 for a computationally intensive scenario, and a batch size of 32 for a communicationally intensive scenario.


.. raw:: html

    <p align="center"><img src="https://user-images.githubusercontent.com/16711681/98315290-bce5ee80-1f8c-11eb-931f-297a99d958ed.png" alt="Benchmark 1" width="400"/><img src="https://user-images.githubusercontent.com/16711681/98315305-c2433900-1f8c-11eb-91b8-1b17f31dce68.png" alt="Benchmark 2" width="400"/></p>


In the figures, the black box represents the ideal linear scaling. It is observed that Bluefog can achieve over 95% scaling efficiency while Horovod reaches around 66% sacling efficiency with batch size 64 on 128 GPUs. For the communicationally intensive scenario with batch size 32, the scaling efficiency gap between Bluefog and Horovod becomes even larger. To 
understand more details about the BlueFog benchmark, checkout our `performance page <https://bluefog-lib.github.io/bluefog/performance.html>`_.

Overview
--------
BlueFog is built with decentralized optimization algorithms. This is fundamentally different from other popular distributed training frameworks, such as DistributedDataParallel provided by PyTorch, Horovod, BytePS, etc. 

In each communication stage, neither the typical star-shaped parameter-server toplogy, nor the pipelined ring-allreduce topology is used. Instead, BlueFog will exploit a virtual and probably dynamic network topology (that can be in any shape) to achieve most communication efficiency.


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


Quick Start
-----------

First, make sure your environment is with ``python>=3.7`` and ``openmpi >= 4.0``.
Then, install Bluefog with: ``pip install --no-cache-dir bluefog`` or
``BLUEFOG_WITH_NCCL=1 pip install bluefog`` if NCCL is supported (``NCCL>=2.7``). Check
the `install_bluefog <https://bluefog-lib.github.io/bluefog/install.html>`_ page if you need more information or other install options.

We provide high-level wrapper for torch optimizer. You just need to modify
the existing script to distributed implementation is wrapping the optimizer
with our ``DistributedNeighborAllreduceOptimizer``,
then run it through ``bfrun``. That is it!

.. code-block:: python

   # Execute Python functions in parallel through
   # bfrun -np 4 python file.py

   import torch 
   import bluefog.torch as bf
   ...
   bf.init()
   optimizer = optim.SGD(model.parameters(), lr=lr * bf.size())
   optimizer = bf.DistributedNeighborAllreduceOptimizer(
      optimizer, model=model
   )
   ...
Previous example is for static topology usage. For dynamic topology case, you need a little bit
more code:

.. code-block:: python
   
  from bluefog.common import topology_util
  ...
  # Same setup code as previous snippets
  dynamic_neighbors_gen = topology_util.GetInnerOuterExpo2DynamicSendRecvRanks(
            bf.size(), local_size=bf.local_size(), self_rank=bf.rank())
  def dynamic_topology_update(epoch, batch_idx):
    send_neighbors, recv_neighbors = next(dynamic_neighbors_gen)
    avg_weight = 1/(len(recv_neighbors) + 1)
    optimizer.send_neighbors = to_neighbors
    optimizer.neighbor_weights = {r: avg_weight for r in recv_neighbors}
    optimizer.self_weight = avg_weight

  # Torch training code
  for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        dynamic_topology_update(epoch, batch_idx)
        ...
        loss.backward()
        optimizer.step()

Check our BlueFog `dynamic topology neighbor averaging <https://bluefog-lib.github.io/bluefog/neighbor_average.html>`_
page to see more on how to control and use topology. See BlueFog `examples`_ folder for full code.


We also provide lots of low-level functions, which you can use those as building
blocks to construct your own distributed training algorithm. The following example
illustrates how to run a simple consensus algorithm through bluefog.

.. code-block:: python

   import torch
   import bluefog.torch as bf

   bf.init()
   x = torch.Tensor([bf.rank()])
   for _ in range(100):
      x = bf.neighbor_allreduce(x)
   print(f"{bf.rank()}: Average value of all ranks is {x}")

Checkout our `API explanation page <https://bluefog-lib.github.io/bluefog/bluefog_ops.html>`_ to see all supported *synchronous* and *asynchronous* features.

The Bluefog source code was based off `Horovod <https://github.com/horovod/horovod>`_ repository. Hence, BlueFog shared lots of common features from Horovod such as `timeline <https://bluefog-lib.github.io/bluefog/timeline.html>`_, tensor-fusion, etc. Here, we want to express our gratitude to the Horovod team. 

Citation
--------
*BlueFog: Make Decentralized Algorithms Practical for Optimization and Deep Learning*, Bluefog Team, To be Appeared in 2020

.. _AWS: https://aws.amazon.com/about-aws/whats-new/2018/12/introducing-amazon-ec2-p3dn-instances-our-most-powerful-gpu-instance-yet/
.. _examples: https://github.com/Bluefog-Lib/bluefog/tree/master/examples
