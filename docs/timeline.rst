Bluefog Timeline
================

Bluefog timeline is used to record all activities occurred during your distributed training 
process. With Bluefog timeline, you can understand the performance of your training 
algorithm, indentify the bottleneck, and then improve it. 

The development of Bluefog timeline is based on the `Horovod timeline`_. Similar to Horovod, 
Bluefog timeline clearly visualizes the start and end of all communication stages between 
agents such as ``allreduce``, ``broadcast``, ``neighbor_allreduce``, ``win_put``, 
``win_accumulate`` and many others. Some of these communication primitives are exclusive 
to Bluefog. 

An enhanced feature of Bluefog timeline is it also visualizes the computation states of each 
agent such as ``forward propogation`` and ``backward propogation``. The visualization of both 
communication and computation will help a better understanding of your training algorithm. 
For example, Bluefog timeline will tell how your computation is in parallel with the 
communication.

Usage
--------------------------
To record a Bluefog timeline, set ``--timeline-filename`` command line argument to the 
location of the timeline file to be created. This will generate a timeline record file
for each agent. For example, the following command ::

    $ bfrun -np 4 --timeline-filename /path/to/timeline_filename python example.py

will generate four timeline files: ``timeline_filename0.json``, ``timeline_filename1.json``, 
``timeline_filename2.json``, and ``timeline_filename3.json``, and each json file is for 
a different agent. You can then load the timeline file into the 
`chrome://tracing`_ facility of the Chrome browser. If the operation ``--timeline-filename``
is not set, the timeline function will be deactivated by default.



An Example
--------------------------
To be added

.. _Horovod timeline:  https://github.com/horovod/horovod/blob/master/docs/timeline.rst
.. _chrome://tracing:  chrome://tracing/