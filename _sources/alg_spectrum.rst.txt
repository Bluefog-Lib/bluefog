The Spectrum of Distributed Machine Learning Algorithm
======================================================

Current machine learning problem typically is associated with
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

    + Sharded PS
    + Hierarchical PS

  2. Peer-to-Peer ----- (Distributed but also decentralized)

    + Ring-AllReduce
    + Neighbor-Collective

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

