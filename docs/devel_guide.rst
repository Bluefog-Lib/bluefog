BlueFog Development Guide
=========================

Please follow the listed steps to start the development.

0. Check your development environment.
--------------------------------------

Due to limited time and the preliminary phase of this project, we didn’t
extensively test our project on the various environments. In order to
avoid some potential pitfall, please make sure that ``python>=3.7`` and
``gcc>=4.0`` and supporting ``std=C++11`` in your development
environment. We recommend using `conda`_ as python environment control.
Also, prefer to use Mac OS or Linux-base system. Lastly, we relied on
the MPI implementation, please install ``open-mpi>=4.0`` [`Download`_
and `Instruction`_]. (MPI-CH may be fine but not tested yet.)

WARNING: Making sure that there is only one MPI implementation is
installed on your machine.

1. Install the package locally
------------------------------

Under the root folder, i.e. ``python`` folder, run the this command
first.

.. code:: bash

   pip install -e . --verbose

The -e means `“editable”`_, so changes you make to files in the bluefog
directory will take effect without reinstalling the package. In
contrast, if you do python setup.py install, files will be copied from
the bluefog directory to a directory of Python packages (often something
like ``/home/{USER}/anaconda3/lib/python3.7/site-packages/bluefog``).
This means that changes you make to files in the bluefog directory will
not have any effect.

2. Build customer C extension
-----------------------------

We heavily relied on the C extension in this project. Unlike python
file, whenever you modified the C files, you have to re-compile it and
generate the shared library out.

.. code:: bash

   python setup.py build_ext -i

where -i means the in-place build. If your environment is fine, it
should be able to generate a file like ``mpi_lib.cpython-37m-darwin.so``
under ``/bluefog/torch`` folder. (You may have different "middle" name
based on your system and enviroment).

3. Run Test
-----------

To check the setup and build is correct or not, run

::

   make test

to see if all tests can pass or not. So far the test is based on python
implementation, in order to test your c extension is built correctly,
please run

.. code:: bash

   BLUEFOG_LOG_LEVEL=debug mpirun -n 2 python bluefog/torch/c_hello_world.py

as well. Currently, we are in the phase of transferring from python
implementation into C++ implementation. We still keep both
implementations. By default, we use C++ implementation. If you want to
use python implementation, just set the environment variable:

::

   export BLUEFOG_PY_IMPL=1

4. Code Style And Lint
----------------------

It is important to keep the code style and lint consistent throughout
the whole project.

For python, we use normal pylint, which is specified in the
``.pylintrc`` file. Python docstring style is `google style`_. It is
recommended to have an editor to run ``pylint`` easily (But do not turn
on format-on-save.) Otherwise, remember to run ``pylint {FILENANE}`` on
your changes.

For C++, we use ``clang-tidy``? I am not very familiar with C++ format.
Right now, we have a simple ``.clang-format`` file. I just use vscode
``clang-format`` plugin.

5. FAQ for Mac users:
---------------------

1. If my default python version is 2, how to set the default python
   version to 3?

**Answer**: alias python=python3

2. If my pytorch is not well installed, how to reinstall pytorch?

**Answer**: Uninstall torch: pip uninstall torch

Install torch: pip3 install torch torchvision

3. I got the following error when executing “BLUEFOG_LOG_LEVEL=debug
   mpirun -n 2 python bluefog/torch/c_hello_world.py”. How to address
   this issue? Error: File “bluefog/torch/c_hello_world.py”, line 36
   print(f“Rank: {rank}, local rank: {local_rank} Size: {size}, local
   size: {local_size}”) ^ SyntaxError: invalid syntax

**Answer**: you should precise python3 using the following command:
BLUEFOG_LOG_LEVEL=debug mpirun -n 2 python3
bluefog/torch/c_hello_world.py

4. If I get the following error when executing “make test” command, how
   to address this issue? Error: Test error: There are not enough slots
   available in the system to satisfy the 4

**Answer**: Reason: there are not enough physical CPU cores (the test
requires 4) in your machine. In order to address this issue, you should
first use “sysctl hw.physicalcpu hw.logicalcpu” command to know the
number of physical CPU cores. Assume that you have 2 physical CPU cores
in your machine, you need to modify 4 in python/Makefile to 2. Then, the
issue is resolved.


.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _Download: https://www.open-mpi.org/software/ompi/v4.0/
.. _Instruction: https://www.open-mpi.org/faq/?category=building#easy-build
.. _“editable”: https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs
.. _google style: http://google.github.io/styleguide/pyguide.html#383-functions-and-methods
