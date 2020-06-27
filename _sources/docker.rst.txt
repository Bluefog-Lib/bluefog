.. _Docker Usage:

Bluefog Docker Usage
=============================

Bluefog provides dockers with all necessary dependency for system environment isolation.
Therefore two types of dockerfiles inside the project.
``dockerfile.cpu`` and ``dockerfile.gpu`` are two files used for actual deployment,
and ``dockerfile.cpu.test`` and ``dockerfile.gpu.test`` are used for internal development.
This document will focus on the usage of ``dockerfile.cpu`` and ``dockerfile.gpu``.
For the dockerfiles for development, the readers can check the Github Wiki for more details.

Installing Docker Image
-----------------------

The docker images can be built from scratch or be obtained from
`Docker Hub <https://hub.docker.com/r/bluefoglib/bluefog>`_.

Downloading Docker Image From Docker Hub
########################################

1. Download docker image with CUDA support:

.. code-block:: bash

    sudo docker pull bluefoglib/bluefog:gpu-0.2.1

2. Download docker image with only CPU support:

.. code-block:: bash

    sudo docker pull bluefoglib/bluefog:cpu-0.2.1

Building Your Own Docker Image
##############################

1. Build docker image with CUDA support:

.. code-block:: bash

    sudo docker build -t bluefog_gpu . -f dockerfile.gpu

2. Build docker image with only CPU support:

.. code-block:: bash

    sudo docker build -t bluefog_cpu . -f dockerfile.cpu

Running Docker Container
------------------------

Here we used the docker images built from scratch as examples.
Please make sure you used the correct docker image name in the following commands,
if you download the docker image from `Docker Hub <https://hub.docker.com/r/bluefoglib/bluefog>`_.

1. Run docker container with CUDA support:

.. code-block:: bash

    sudo docker run --privileged -it --gpus all --name bluefog_gpu_deploy --network=host -v /mnt/share/ssh:/root/.ssh bluefog_gpu:latest

2. Run docker container with only CPU support:

.. code-block:: bash

    sudo docker run --privileged -it --name bluefog_cpu_deploy --network=host -v /mnt/share/ssh:/root/.ssh bluefog_cpu:latest

3. Clean up docker system after running:

.. code-block:: bash

    sudo docker system prune

Nvidia Container Runtime
########################

The following error may pop up when running a docker container with GPUs.

.. code-block:: bash

    docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].

In order to properly run docker with GPUs,
Nvidia container runtime needs to be installed using following commands for Ubuntu.
Furthermore, the GPU driver is also required.

.. code-block:: bash

    curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
        sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
    sudo apt-get update
    sudo apt-get install nvidia-container-runtime
    sudo service docker restart

More details can be found on
`https://github.com/NVIDIA/nvidia-container-runtime <https://github.com/NVIDIA/nvidia-container-runtime>`_
and `https://nvidia.github.io/nvidia-container-runtime <https://nvidia.github.io/nvidia-container-runtime>`_.

Running Examples in Docker Containers
-------------------------------------

The docker images have already included a few examples for the Bluefog library and some unittests for users.

1. UnitTest in docker container

.. code-block:: bash

    ./run_unittest.sh

2. Examples in docker container

.. code-block:: bash

    bfrun -np 4 python examples/pytorch_average_consensus.py
