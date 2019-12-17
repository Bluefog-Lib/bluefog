MPIRUN = mpirun
PYTEST = pytest

test: test_torch
test_torch: test_torch_basic test_torch_ops
test_tensorflow: test_tensorflow_basic test_tensorflow_ops

clean: clean_build clean_so

.PHONY: test_torch_basic
test_torch_basic:
	${PYTEST} ./test/torch_basics_test.py && ${MPIRUN} -np 4 ${PYTEST} ./test/torch_basics_test.py

.PHONY: test_torch_ops
test_torch_ops:
	${MPIRUN} -np 4 ${PYTEST} ./test/torch_ops_test.py

.PHONY: test_tensorflow_basic
test_tensorflow_basic:
	${PYTEST} ./test/tensorflow_basics_test.py && ${MPIRUN} -np 4 ${PYTEST} ./test/tensorflow_basics_test.py

.PHONY: test_tensorflow_ops
test_tensorflow_ops:
	${MPIRUN} -np 4 ${PYTEST} ./test/tensorflow_ops_test.py

.PHONY: clean_build
clean_build:
	rm -R build

.PHONY: clean_so
clean_so:
	rm ./bluefog/torch/mpi_lib.*.so; rm ./bluefog/tensorflow/mpi_lib.*.so
