Bluefog
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

Performance
-----------
Below is the charts representing the performance of BlueFog that was done on ResNet50 benchmark. Each machine has 8 V100 GPUs (64GB memory) with NVLink-enabled and inter-connected communication speed is 25Gbps. This is the same hardware setup you can get on AWS_. We test the scaling efficiency on 64 batch size, representing computation  intensive scenario, and 32 batch size case for communication intensive.

.. raw:: html

    <p align="center"><img src="https://user-images.githubusercontent.com/16711681/97819514-cf46ec00-1c5d-11eb-933e-459783d974a6.png" alt="Benchmark 1" width="400"/><img src="https://user-images.githubusercontent.com/16711681/97819502-c6eeb100-1c5d-11eb-9930-065cdd48818d.png" alt="Benchmark 2" width="400"/></p>

where black box represents the idea linear scaling. We can see bluefog can achieve over 95% scaling efficiency while Horovod is around 78% sacling efficiency under 64 batchsize. For more communication intensive like 32 batch size, the scaling efficiency between Bluefgo and Horovod becomes even larger. To 
understand more details about BlueFog Benchmark, checkout our performance page.

Overview
--------

Bluefog is a distributed training framework for PyTorch based
on diffusion/consensus-type algorithm.
The goal of Bluefog is to make distributed and decentralized machine learning fast,
fault-tolerant, friendly to heterogeneous environment, and easy to use.

The most distinguishable feature of Bluefog compared with other popular distributed training frameworks, such as 
DistributedDataParallel provided by pytorch, Horovod, BytePS, etc., is that our core implementation rooted on the idea
that we introduce the virtual topology into the multiple processes and 

.. math::

     LOCAL_AVG(param - lr*grad_{k}) ==> param - lr*GLOBAL_AVG(grad_{k})) as algorithm keep iterating

where local average is defined based on the connection in the virtual topology. We support both **static** topology
and **dynamic** topology usages. Among most topologies, we find the dynamic Exponential-2 graph can achieve the best performance
if the number of processes is the power of 2 such as 4, 32, 128 processes. Exponential-2 graph is defined as each process only 
communicates with neighbors that are  2^0, 2^1, ..., 2^t hops away. Dynamic means all processes select
one neighbor only in one iteration and select next neighbor in next iteration as illustrated in the figure:

.. raw:: html

    <p align="center"><img src="https://user-images.githubusercontent.com/16711681/97928035-04654400-1d1b-11eb-91d2-2da890b4522e.png" alt="one-peer-exp2" width="650"/></p>

Under this scenario, the communcation cost for each iteration is only one unit delay, one standard parameter size to transmit and no communication conflict, which
is better than ring-allreduce promised. As for loss and accuracy guarantee, please check out our theoratical paper.

Quick Start
-----------

First, make sure your environment is with ``python>=3.7`` and ``openmpi >= 4.0``.
Then, install Bluefog with: ``pip install --no-cache-dir bluefog`` or
``BLUEFOG_WITH_NCCL=1 pip install bluefog`` if NCCL is supported (NCCL>=2.7). Check
the ``install_bluefog`` page if you need more information or other install options.

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
Check our BlueFog Distributed Optimizer Guide to understand how our distributed optimizer 
works and which distributed optimizer fits your requirement the best.

We also provide lots of low-level functions, which you can use those as building
blocks to construct your own distributed trainning algorithm. The following example
illustrates how to run a simple consensus algorithm through bluefog.

.. code-block:: python

   import torch
   import bluefog.torch as bf

   bf.init()
   x = torch.Tensor([bf.rank()])
   for _ in range(100):
      x = bf.neighbor_allreduce(x)
   print(f"{bf.rank()}: Average value of all ranks is {x}")

Checkout our ops explanation page to see all supported *synchronous* and *asynchronous* features.


Citation
--------
*BlueFog: Make Decentralized Algorithms Practical for Optimization and Deep Learning*, Bluefog Team, To be Appeared in 2020

.. _AWS: https://aws.amazon.com/about-aws/whats-new/2018/12/introducing-amazon-ec2-p3dn-instances-our-most-powerful-gpu-instance-yet/
