The Structure of Codebase
=========================

One the top level, the structure of this codebase looks like:

.. code:: bash

   ├── bluefog
   ├── docs
   ├── examples
   ├── test
   ├── setup.py
   └── ...

Main project code is under ``bluefog`` folder, ``examples`` folder
contains the usage code for users, (served as e2e test as well) and
``test`` folder for test of course. ``docs`` folder contains the
document including API, docker usagem, algorithm explaination, development etc.
Overall, our repository shares a very similar organization like
`horovod`_ repository.


.. _horovod: https://github.com/horovod/horovod

