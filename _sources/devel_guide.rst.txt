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

3. Run Unit Test
----------------

To check the setup and build is correct or not, run

::

   make test

to see if all tests can pass or not. The all test command is defined in the
Makefile under root folder. To see more details, you can manually run the 
following command:

.. code:: bash

   BLUEFOG_LOG_LEVEL=debug mpirun -n 2 python test/torch_ops_test.py

4. Continuous integration and End-to-End test
---------------------------------------------
We use `travis`_ as our continuous integration test. Right now, it has
merely a few functionality. (Unit test running on LINUX + MPICH + python 3.7).
We plan to test more on multiple os platforms (os + linux), python
versions (3.5+),  MPI venders (MPICH and OpenMPI), CPU + GPU, docker
images, etc.

Bluefog is a library oriented project. So there is no real end-to-end for it.
But we can run several examples, which will utilize most functionalities
provided by Bluefog. As long as that examples runs correctly, it can be
considered as passing the end-to-end test. For example:

.. code:: bash

   mpirun -n 4 python examples/pytorch_mnist.py --epoch=5


5. Code Style And Lint
----------------------

It is important to keep the code style and lint consistent throughout
the whole project.

For python, we use normal pylint, which is specified in the
``.pylintrc`` file. Python docstring style is `google style`_. It is
recommended to have an editor to run ``pylint`` easily (But do not turn
on format-on-save.) Otherwise, remember to run ``pylint {FILENANE}`` on
your changes.

For C++, we use ``clang-tidy``. We follow the google style for C++ as well.


.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _Download: https://www.open-mpi.org/software/ompi/v4.0/
.. _Instruction: https://www.open-mpi.org/faq/?category=building#easy-build
.. _“editable”: https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs
.. _google style: http://google.github.io/styleguide/pyguide.html
.. _travis: https://travis-ci.com/ybc1991/bluefog
