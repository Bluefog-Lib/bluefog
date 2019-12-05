#!/bin/bash

horovodrun -n 4 -H localhost:4 python horovod_gan.py --epoch=40
mpirun -n 4 python as_gan_mpi.py --epoch=40