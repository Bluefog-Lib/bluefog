#!/bin/bash
# Assume running under the root directory of Bluefog with `test` folder
NUM_PROC=4
MPIRUN="mpirun -np $NUM_PROC --allow-run-as-root"
PYTEST="pytest -s -vv"

$PYTEST ./test/torch_basics_test.py
$MPIRUN $PYTEST ./test/torch_basics_test.py
$MPIRUN $PYTEST ./test/torch_ops_test.py
$MPIRUN $PYTEST ./test/torch_win_ops_test.py
