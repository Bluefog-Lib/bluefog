Running Bluefog Through bfrun
=============================

The usage of ``bfrun`` is almost same as using ``mpirun`` since ``bfrun`` is just a thin wrapper over
``mpirun`` for your convenience.
Check your MPI documentation for arguments to the `mpirun`
command on your system or you can type ``bfrun -h`` to check all flags it supports.

Typically one GPU will be allocated per process, so if a server has 4 GPUs, you would run 4 processes. In `bfrun`,
the number of processes is specified with the `-np` flag.

1. To run on a machine with 4 GPUs:

.. code-block:: bash

    bfrun -np 4 python train.py


2. To run on 4 machines with 4 GPUs each:

.. code-block:: bash

    bfrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py


Failures due to SSH issues
--------------------------

The host where `bfrun` is executed must be able to SSH to all other hosts without any prompts.

If `bfrun` fails with permission error, verify that you can ssh to every other server without entering a password or
answering questions like this:

::

    The authenticity of host '<hostname> (<ip address>)' can't be established.
    RSA key fingerprint is xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx.
    Are you sure you want to continue connecting (yes/no)?


To learn more about setting up passwordless authentication, see `this page`_.

.. _this page: http://www.linuxproblem.org/art_9.html

