MPIRUN = mpirun
PYTEST = pytest

test: test_torch_basic test_torch_ops

test_torch_basic:
	${PYTEST} ./test/torch_basics_test.py && ${MPIRUN} --allow-run-as-root -n 4 ${PYTEST} ./test/torch_basics_test.py

test_torch_ops:
	${MPIRUN} --allow-run-as-root -n 4 ${PYTEST} ./test/torch_ops_test.py

clean_build:
	rm -R build

clean_so:
	rm ./bluefog/torch/mpi_lib.*.so

clean_all: clean_build clean_so
