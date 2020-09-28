NUM_PROC ?= 4
$(info $(shell mpirun --version))
ifeq ($(findstring Open MPI, $(shell mpirun --version)), Open MPI)
  EXTRA_MPI_FLAG = -mca ^openib --allow-run-as-root
else
  EXTRA_MPI_FLAG = 
endif

MPIRUN = mpirun -np ${NUM_PROC} ${EXTRA_MPI_FLAG}
PYTEST = pytest -s -vv
MPICH_NOT_EXIST = $(shell which mpichversion)

.PHONY: build
build:
	python setup.py build_ext -i

test: test_torch
test_torch: test_torch_basic test_torch_ops test_torch_win_ops
test_tensorflow: test_tensorflow_basic test_tensorflow_ops
test_all: test_torch test_tensorflow

clean: clean_build clean_so

.PHONY: test_torch_basic
test_torch_basic:
	${PYTEST} ./test/torch_basics_test.py && ${MPIRUN} ${PYTEST} ./test/torch_basics_test.py

.PHONY: test_torch_ops
test_torch_ops:
	${MPIRUN} ${PYTEST} ./test/torch_ops_test.py

.PHONY: test_timeline
test_timeline:
	${MPIRUN} ${PYTEST} ./test/timeline_test.py

.PHONY: test_torch_win_ops
test_torch_win_ops:
	${MPIRUN} ${PYTEST} ./test/torch_win_ops_test.py

.PHONY: test_tensorflow_basic
test_tensorflow_basic:
	${PYTEST} ./test/tensorflow_basics_test.py && ${MPIRUN} ${PYTEST} ./test/tensorflow_basics_test.py

.PHONY: test_tensorflow_ops
test_tensorflow_ops:
	${MPIRUN} ${PYTEST} ./test/tensorflow_ops_test.py

.PHONY: clean_build
clean_build:
	rm -R build

.PHONY: clean_so
clean_so:
	rm ./bluefog/torch/mpi_lib.*.so
