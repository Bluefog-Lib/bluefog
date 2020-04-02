#!/bin/bash

set -e
set -x

function build_one()
{
    device=$1

    tag=bluefog-${device}:$(date +%Y%m%d)
    docker build -f Dockerfile.${device}.test -t ${tag} --no-cache .
}

# clear upstream images, ok to fail if images do not exist
docker rmi $(cat Dockerfile.gpu.test | grep FROM | awk '{print $2}') || true
docker rmi $(cat Dockerfile.cpu.test | grep FROM | awk '{print $2}') || true

# build cpu and gpu images
build_one gpu
build_one cpu

# print recent images
docker images $(docker images | grep bluefog | awk '{print $1}')
