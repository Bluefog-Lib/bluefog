Bluefog Docker Usage
=============================

1. Build docker image with CUDA support:

.. code-block:: bash

    sudo docker build -t bluefog_gpu . -f dockerfile.gpu

2. Build docker image with only CPU support:

.. code-block:: bash

    sudo docker build -t bluefog_cpu . -f dockerfile.cpu

1. Run docker container with CUDA support:

.. code-block:: bash

    sudo docker run --privileged -it --gpus all --name bluefog_gpu_deploy --network=host -v /mnt/share/ssh:/root/.ssh bluefog_gpu:latest

2. Run docker container with only CPU support:

.. code-block:: bash

    sudo docker run --privileged -it --name bluefog_cpu_deploy --network=host -v /mnt/share/ssh:/root/.ssh bluefog_cpu:latest

3. Close docker container with CUDA support:

.. code-block:: bash

    sudo docker container rm bluefog_gpu_deploy

4. Close docker container with only CPU support:

.. code-block:: bash

    sudo docker container rm bluefog_cpu_deploy

1. UnitTest in docker container

.. code-block:: bash

    bfrun -np 4 pytest -s torch_ops_test.py

2. Examples in docker container

.. code-block:: bash

    bfrun -np 4 python pytorch_average_consensus.py
