MPIRUN = mpirun
PYTEST = pytest

test: test_torch_basic test_torch_ops

test_torch_basic:
	${PYTEST} ./test/torch_basics_test.py && ${MPIRUN} -np 4 ${PYTEST} ./test/torch_basics_test.py

test_torch_ops:
	${MPIRUN} -np 4 ${PYTEST} ./test/torch_ops_test.py

test_tensorflow_basic:
	${PYTEST} ./test/tensorflow_basics_test.py && ${MPIRUN} -np 4 ${PYTEST} ./test/tensorflow_basics_test.py

test_tensorflow_ops:
	${MPIRUN} -np 4 ${PYTEST} ./test/tensorflow_ops_test.py

clean_build:
	rm -R build

clean_so:
	rm ./bluefog/torch/mpi_lib.*.so; rm ./bluefog/tensorflow/mpi_lib.*.so

clean_all: clean_build clean_so
