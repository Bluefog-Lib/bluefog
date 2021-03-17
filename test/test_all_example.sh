#!/bin/bash
NUM_PROC=4
BFRUN="bfrun -np ${NUM_PROC}"
RUN_DIR="$( pwd )"
EXAMPLE_DIR="$( cd "$( dirname "$0" )" && pwd )/../examples"

die() { echo >&2 -e "\nERROR: $@\n"; exit 1; }
check() {
    timeout 2m "$@" >>/dev/null 2>&1;
    local exit_code=$?;
    [ $exit_code -eq 0 ] \
        && echo "Command [$*] succeed" \
        || die "Command [$*] failed with error code $exit_code";
}

# check GPU exists
nvidia-smi >>/dev/null 2>&1
gpu_exit_code=$?
if [[ $gpu_exit_code -eq 0 ]]; then
    isgpu=1
else
    isgpu=0
fi

if [[ $isgpu -eq 1 ]]; then
    echo "GPU Detected"
else
    echo "No GPU Detected."
fi

# PyTorch Average Concensus Cases
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_average_consensus.py
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_average_consensus.py --enable-dynamic-topology
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_average_consensus.py --asynchronous-mode
if [[ $isgpu -eq 1 ]]; then
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_average_consensus.py --no-cuda
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_average_consensus.py --no-cuda --enable-dynamic-topology
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_average_consensus.py --no-cuda --asynchronous-mode
fi

# PyTorch Optimization Cases
[ -f "${RUN_DIR}/plot.png" ] && EXIST_PLOT_PNG=1 || EXIST_PLOT_PNG=0
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_optimization.py --method=diffusion
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_optimization.py --method=exact_diffusion
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_optimization.py --method=gradient_tracking
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_optimization.py --method=push_diging
[ "${EXIST_PLOT_PNG}" == 0 ] && rm -f ${RUN_DIR}/plot.png

# PyTorch MNIST Cases
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --dist-optimizer=gradient_allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --dist-optimizer=allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --dist-optimizer=allreduce --atc-style
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --dist-optimizer=neighbor_allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --dist-optimizer=neighbor_allreduce --disable-dynamic-topology
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --dist-optimizer=neighbor_allreduce --atc-style
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --dist-optimizer=neighbor_allreduce --atc-style --disable-dynamic-topology
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --dist-optimizer=win_put
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --dist-optimizer=win_put --disable-dynamic-topology
if [[ $isgpu -eq 1 ]]; then
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --no-cuda --dist-optimizer=gradient_allreduce
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --no-cuda --dist-optimizer=allreduce
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --no-cuda --dist-optimizer=allreduce --atc-style
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --no-cuda --dist-optimizer=neighbor_allreduce
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --no-cuda --dist-optimizer=neighbor_allreduce --disable-dynamic-topology
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --no-cuda --dist-optimizer=neighbor_allreduce --atc-style
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --no-cuda --dist-optimizer=neighbor_allreduce --atc-style --disable-dynamic-topology
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --no-cuda --dist-optimizer=win_put
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_mnist.py --epochs=1 --no-cuda --dist-optimizer=win_put --disable-dynamic-topology
fi

# PyTorch Benchmark Cases
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --dist-optimizer=gradient_allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --dist-optimizer=allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --dist-optimizer=allreduce --atc-style
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --dist-optimizer=neighbor_allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --dist-optimizer=neighbor_allreduce --disable-dynamic-topology
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --dist-optimizer=neighbor_allreduce --atc-style
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --dist-optimizer=neighbor_allreduce --atc-style --disable-dynamic-topology
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --dist-optimizer=win_put
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --dist-optimizer=win_put --disable-dynamic-topology
if [[ $isgpu -eq 1 ]]; then
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --no-cuda --dist-optimizer=gradient_allreduce
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --no-cuda --dist-optimizer=allreduce
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --no-cuda --dist-optimizer=allreduce --atc-style
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --no-cuda --dist-optimizer=neighbor_allreduce
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --no-cuda --dist-optimizer=neighbor_allreduce --disable-dynamic-topology
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --no-cuda --dist-optimizer=neighbor_allreduce --atc-style
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --no-cuda --dist-optimizer=neighbor_allreduce --atc-style --disable-dynamic-topology
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --no-cuda --dist-optimizer=win_put
    check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_benchmark.py --model=lenet --num-iters=1 --no-cuda --dist-optimizer=win_put --disable-dynamic-topology
fi

# PyTorch ResNet Cases
[ -d "${RUN_DIR}/logs" ] && EXIST_LOGS_DIR=1 || EXIST_LOGS_DIR=0
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --dist-optimizer=gradient_allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --dist-optimizer=allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --dist-optimizer=allreduce --atc-style
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --dist-optimizer=neighbor_allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --dist-optimizer=neighbor_allreduce --disable-dynamic-topology
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --dist-optimizer=neighbor_allreduce --atc-style
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --dist-optimizer=neighbor_allreduce --atc-style --disable-dynamic-topology
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --dist-optimizer=win_put
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --dist-optimizer=win_put --disable-dynamic-topology
if [[ $isgpu -eq 1 ]]; then
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --no-cuda --dist-optimizer=gradient_allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --no-cuda --dist-optimizer=allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --no-cuda --dist-optimizer=allreduce --atc-style
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --no-cuda --dist-optimizer=neighbor_allreduce
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --no-cuda --dist-optimizer=neighbor_allreduce --disable-dynamic-topology
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --no-cuda --dist-optimizer=neighbor_allreduce --atc-style
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --no-cuda --dist-optimizer=neighbor_allreduce --atc-style --disable-dynamic-topology
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --no-cuda --dist-optimizer=win_put
check ${BFRUN} python ${EXAMPLE_DIR}/pytorch_resnet.py --model=squeezenet1_0 --epochs=1 --no-cuda --dist-optimizer=win_put --disable-dynamic-topology
fi
[ "${EXIST_LOGS_DIR}" == 0 ] && rm -rf ${RUN_DIR}/logs