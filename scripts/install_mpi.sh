#!/bin/sh
set -e
case `uname` in
Linux)
  case $1 in
    mpich) set -x;
      sudo apt-get install -y -q mpich libmpich-dev
      ;;
    openmpi) set -x;
      mkdir /tmp/openmpi && \
      cd /tmp/openmpi && \
      wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
      tar zxf openmpi-4.0.0.tar.gz && \
      cd openmpi-4.0.0 && \
      ./configure --enable-orterun-prefix-by-default && \
      make -j $(nproc) all && \
      make install && \
      ldconfig && \
      rm -rf /tmp/openmpi
      ;;
    *)
      echo "Unknown MPI implementation:" $1
      exit 1
      ;;
  esac
  ;;
Darwin)
  case $1 in
    mpich) set -x;
      brew install mpich
      ;;
    openmpi) set -x;
      mkdir /tmp/openmpi && \
      cd /tmp/openmpi && \
      wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
      tar zxf openmpi-4.0.0.tar.gz && \
      cd openmpi-4.0.0 && \
      ./configure --enable-orterun-prefix-by-default && \
      make -j $(nproc) all && \
      make install && \
      ldconfig && \
      rm -rf /tmp/openmpi
      ;;
    *)
      echo "Unknown MPI implementation:" $1
      exit 1
      ;;
  esac
  ;;
esac
