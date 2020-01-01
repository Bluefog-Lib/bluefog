Bluefog
=======

.. image:: https://travis-ci.com/ybc1991/bluefog.svg?token=me5bQ3zp2qcSz5D3yVNC&branch=master
    :target: https://travis-ci.com/ybc1991/bluefog

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :alt: License

.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    
Overview
--------
Bluefog is a distributed training framework for Tensorflow and PyTorch
based on diffusion/consensus-type algorithm. The goal of Bluefog is to make
distributed machine learning fast and fault-tolerant in the heterogeneous
environment and users are easy to set up and run experiments without worrying
too many low-level details.

REPOSITORY IS STILL A WORK IN PROGRESS.

Philosophy
----------
There are already lots of well-designed and production-level distributed
machine learning algorithms, libraries, frameworks, or tools.
What is the main different between Bluefog project and others?  
Why can Bluefog outperform others? 
Which scenario is more suitable for Bluefog?


Before answering above questions, *Demystifying 
Parallel and Distributed Deep Learning* [1]_ paper has a great conclusion:

::

 The world of deep learning is brimming with concurrency. Even if an aspect
 is sequential, its consistency requirements can be reduced, due to the
 robustness of nonlinear optimization, to increase concurrency while 
 still attaining reasonable accuracy, if no better.
 
The main philosophy is we can sacrifice the consistency or sequential requirement
to gain faster trainning speed, more robust system, and more friendly to the
heterogeneous enviroment.

[Add more technique details here.]

To comparison with the other algorithms/libraries, we need to demystifying several
importance perspectives viewing a distributed machine learning library first.

The Spectrum of Distributed Machine Learning Algorithm
------------------------------------------------------
Current machine learning problem is always associated with
the large scale of dataset and highly complexity of the model.
This provides us lots of ascpects and options to design the algorithm 
so that the multi-core CPU/GPU in the distributed computation system
can be fully exploited. We list the most relevant and important 
considerations here.

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

* From the aspect of communication architecture:

  1. Parameter Server(PS) ---- (Distributed but still centralized)

    - Sharded PS 
    - Hierarchical PS

  2. Peer-to-Peer ----- (Distributed but also decentralized)

    - Ring-AllReduce
    - Neighbor-Collective

  Apparently, Bluefog project belongs to the peer-to-peer model. Multiple nodes/machines
  will distributedly and no centralized node will gather all the informations.

* From the aspect of parameter consistency:

  In the distributed learning system, parameter consistency means the similarity
  between the parameter stored in the local machine. We list five typical 
  algorithms from strongest consistency to weakest consistency.

  1. Model Replication
  2. Delayed Updating (like asynchronous algorithm.)
  3. Model Averaging
  4. Ensemble Learning

  Bluefog project focused on the asynchronous training through the
  diffusion/consensus algorithm, which is one kind of
  model averaging algorithms. The parameter learned in different nodes 
  through the Bluefog algorithm are slightly different. But unlike 
  ensemble learning, all nodes are highly similar.

* From the aspect of updating synchronization:

  1. Synchronous updating
  2. Stale-Synchronous updating
  3. Asynchronous updating
  
  Typically, the more "asynchronous" updating, the faster on the training. However, 
  we will loss the parameter consistency.

* From the aspect of information fusion:

  1. Averaging over the gradients
  2. Averaging over the parameter/iterates
  3. Averaging over the dual variable
  
  Most strong consistency algorithm is averaging over the gradients. However, Bluefog project
  is averaging over the parameter directly. One advantage of averaging over the parameter is
  the resilient on the noise or error. Also, noticing these three methods are not exclusive. 

* From the aspect of reducing communication cost:

  1. Temporal compression (Fine- vs Coarse-Grained Fusion)
  2. Spatial compression (Sparse/Sliced tensors)
  3. Btye compression (Quantization)
  4. Neighbor compression (Selecting less neighbors)

  We don't have any implementation to support it yet. We do plan to support it in
  the future.
  
.. [1] Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis (https://arxiv.org/abs/1802.09941)

