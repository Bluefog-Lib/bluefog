.. Bluefog documentation master file, created by
   sphinx-quickstart on Mon Dec  2 20:20:32 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bluefog
=======

.. image:: _static/bluefog_logo.png
   :width: 350
   :align: center
   :alt: Bluefog Logo

Bluefog is a distributed training framework for PyTorch based
on diffusion/consensus-type algorithm.
The goal of Bluefog is to make distributed machine learning fast,
fault-tolerant, friendly to heterogeneuous environment, and easy to use.

.. Important::

   Although most torch-based APIs perform well, this repository is still 
   in the early stage of development and more features are waiting to be implemented.
   If you are interested, you are more than welcome to join us and contribute this project!

Quick Start
-----------

First, make sure your environment has ``python>=3.7`` and `openmpi`_ >= 4.0.
Then, install Bluefog with: ``pip install bluefog``.  We provide high-level wrapper for optimizer. 
Probably, the only thing you need to modify
the existing script is wrapping the optimizer with our ``DistributedBluefogOptimizer``,
then run it through ``bfrun``. That is it!

.. code-block:: python

   # Execute Python functions in parallel through
   # bfrun -np 4 python file.py

   import torch 
   import bluefog.torch as bf
   ...
   bf.init()
   optimizer = optim.SGD(model.parameters(), lr=lr * bf.size())
   optimizer = bf.DistributedBluefogOptimizer(
      optimizer, named_parameters=model.named_parameters()
   )
   ...

We also provide low-level functions, which you can use those as building
blocks to construct your own distributed trainning algorithm. The following example
illustrate how to use bluefog run a simple consensus algorithm.

.. code-block:: python

   import torch
   import bluefog.torch as bf

   bf.init()
   x = torch.Tensor([bf.rank()])
   for _ in range(100):
      x = bf.neighbor_allreduce(x)
   print(f"{bf.rank()}: Average value of all ranks is {x}")

One main feature of Bluefog is that we leverage the One-sided Communication of MPI
to build a real decentralized and asynchronized algorithms. The following code illustrate
how to use Bluefog to implement an asynchronized push-sum consensus algorithm.

.. code-block:: python

   import torch
   import bluefog.torch as bf
   from bluefog.common import topology_util

   bf.init()

   # Setup the topology for communication
   bf.set_topology(topology_util.PowerTwoRingGraph(bf.size()))
   outdegree = len(bf.out_neighbor_ranks())
   indegree = len(bf.in_neighbor_ranks())

   # Create the buffer for neighbors.
   x = torch.Tensor([bf.rank(), 1.0])
   bf.win_create(x, name="x_buff", zero_init=True)

   for _ in range(100):
      bf.win_accumulate(
         x, name="x_buff",
         dst_weights={rank: 1.0 / (outdegree + 1)
                      for rank in bf.out_neighbor_ranks()},
         require_mutex=True)
      x.div_(1+outdegree)
      bf.win_sync_then_collect(name="x_buff")

   bf.barrier()
   # Do not forget to sync at last!
   bf.win_sync_then_collect(name="x_buff")
   print(f"{bf.rank()}: Average value of all ranks is {x[0]/x[-1]}")

Please explore our `examples`_ folder to see more examples about
how to use bluefoge implemented deep learning trainning and distributed
optimization algorithm quickly and easily.

.. toctree::
   :maxdepth: 1
   :caption: INSTALLATION

   Installing Bluefog <install>


.. toctree::
   :maxdepth: 2
   :caption: API

   Bluefog Common API <common_api>
   Bluefog Torch API <torch_api>
   Bluefog Tensorflow API <tensorflow_api>

.. toctree::
   :maxdepth: 2
   :caption: More Information

   Bluefog Timeline <timeline>
   Bluefog WinOps Explanation <bluefog_winops>
   Codebase Structure <code_structure>
   Development Guide <devel_guide>
   Spectrum of Machine Learning Algorithm<alg_spectrum>

.. _openmpi: https://www.open-mpi.org/
.. _examples: https://github.com/ybc1991/bluefog/tree/master/examples