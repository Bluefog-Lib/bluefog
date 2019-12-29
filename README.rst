Bluefog
=======

.. image:: https://travis-ci.com/ybc1991/bluefog.svg?token=me5bQ3zp2qcSz5D3yVNC&branch=master
    :target: https://travis-ci.com/ybc1991/bluefog

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :alt: License

Overview
--------
Bluefog is a distributed training framework for Tensorflow and PyTorch
based on diffusion/consensus-type algorithm. The goal of Bluefog is to make
distributed machine learning fast and fault-tolerant in the heterogeneous
environment and users are easy to set up and run experiments without worrying
too many low-level details.

REPOSITORY IS STILL A WORK IN PROGRESS.

The Spectrum of Distributed Machine Learning Algorithm
------------------------------------------------------
The nature of current machine learning problem typically is associated with
the large scale of dataset and highly complexity of the model.
This provides us lots of ascpects and options to design so that we can fully
exploit the multi-core CPU/GPU in the distributed computation system.
We list the most relevant and important considerations here.

* From the aspect of data and modeling concurrency:

  1. Data Parallelism
  2. Model Parallelism
  3. Pipeline Parallelism

  The above three techniques are not exclusive to each other. For example,
  tensorflow allow users can utilize all three techniques at the same time.
  The Bluefog project focused on Data Parallelism.
  One reason, of course, is the base algorithm derived based on the assumption
  that the dataset is distributed over different nodes. But, more importantly, 
  among these three techniques, data parallelism is the most popular approaches
  thanks to its excellent scalability and flexibility on almost any model.

* From the aspect of parameter consistency:

  In the distributed learning system, parameter consistency means the similarity
  between the parameter stored in the local machine. We list five typical 
  algorithms from strongest consistency to weakest consistency.

  1. Synchronous SGD
  2. Stale-Synchronous SGD
  3. Asynchronous SGD
  4. Model Averaging
  5. Ensemble Learning

  Bluefog project focused on the asynchronous training through the
  diffusion/consensus algorithm. Diffusion/consensus algorithm is one kind of
  model averaging algorithms. Therefore, Bluefog project belongs to level 3-4.

* From the aspect of communication architecture:

  1. Parameter Server(PS) ---- (Distributed but still centralized)

    - Sharded PS 
    - Hierarchical PS

  2. Peer-to-Peer ----- (Distributed but also decentralized)

    - Ring-AllReduce
    - Neighbor-Collective

  Apparently, Bluefog project belongs to the peer-to-peer model. Multiple nodes/machines
  will distributedly and no centralized node will gather all the informations.


* From the aspect of communication cost:

  1. Temporal compression (Fine- vs Coarse-Grained Fusion)
  2. Spatial compression (Sparse/Sliced tensors)
  3. Btye compression (Quantization)
  4. Neighbor compression (Selecting less neighbors)

  We don't have any implementation to support it yet. We do plan to support it in
  the future.
