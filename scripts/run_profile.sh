#!/bin/bash
now=$(date +"%T")

mpirun -np 4 python -m cProfile -o ./.profile/bf_program.prof as_gan_mpi.py --skip_tensorboard --epoch=2

# Run it on local machine
# snakeviz ./.profile/program.prof