Bluefog Timeline
================

Bluefog timeline is used to record all activities occurred during your distributed training 
process. With Bluefog timeline, you can understand the performance of your training 
algorithm, indentify the bottleneck, and then improve it. 

The development of Bluefog timeline is based on the `Horovod timeline`_. Similar to Horovod, 
Bluefog timeline clearly visualizes the start and end of all communication stages between 
agents such as `allreduce`, `broadcast`, `neighbor_allreduce`, `win_put`, `win_accumulate` 
and many others. Some of these communication primitives are exclusive to Bluefog. 

An enhanced feature of Bluefog timeline is it also visualizes the computation states of each 
agent such as `forward` and `backward propogation`. The visualization of both communication and
computation will help a better understanding of your training algorithm. For example, Bluefog
timeline will tell how your computation is in parallel with the communication.

Usage
--------------------------
To be added

An Example
--------------------------

.. _Horovod timeline:  https://github.com/horovod/horovod/blob/master/docs/timeline.rst