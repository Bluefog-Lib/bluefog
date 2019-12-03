The Structure of Codebase
=========================

The structure of this codebase is a little messy due to historical
reasons.

One the top level, it looks like this

.. code:: bash

   ├── IndependentApp
   ├── __tests__
   ├── android
   ├── app
   ├── ios
   ├── python
   └── ...

where ``IndependentApp, android, ios``, etc are the old project that
related to APP idea, which can be ignored for now (2019Q2).

The recent *bluefog* project is all located under ``python`` folder.
Under it, the structure looks like:

.. code:: bash

   ├── bluefog
   ├── docs
   ├── examples
   ├── obsolete
   ├── setup.py
   ├── test
   └── ...

The obsolete folder contains some old experimental code, which can be
ignored (If you want to put some scratch or random test and keep on
github for a while, this is the place). Main project code is under
``bluefog`` folder, ``examples`` folder contains the usage code for
users, (served as e2e test as well) and ``test`` folder for test of
course. Overall, our repository shares a very similar organization like
`horovod`_ repository.

.. _horovod: https://github.com/horovod/horovod

