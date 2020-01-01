#!/bin/bash
now=$(date +"%T")
mnist=./examples/pytorch_mnist.py
prof_file=./examples/.profile/mnist.prof
mkdir ./examples/.profile

mpirun -np 4 python -m cProfile -o  $prof_file $mnist --epoch=3

# Run it on local machine
snakeviz $prof_file