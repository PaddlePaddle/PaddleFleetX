# dgc-lib
Deep gradient compression lib.
## Introduction
Base on deep gradient compression paper.
[DGC paper](https://arxiv.org/pdf/1712.01887.pdf).
## Requirements
dgc-lib requires CUDA and NCCL head.
## Build
To build the static library :
```shell
$ make -j src.staticlib
# or default
$ make -j
```
To build the shared library :
```shell
$ make -j src.lib
```
If CUDA or NCCL is not installed in the default path, you can build with :
```shell
$ make CUDA_HOME=<path to cuda install> NCCL_INCLUDE=<path to nccl include>
```
dgc-lib will be compiled in `build/` unless `BUILD_PREFIX` is set.
## Install
```shell
$ make install
```
dgc-lib will be installed in `install/` unless `INSTALL_PREFIX` is set.
## Tests
dgc-test requires CUDA、NCCL and MPI.
```shell
$ make -j test
```
If CUDA 、NCCL or MPI is not installed in the default path, you can build with :
```shell
$ make -j test CUDA_HOME=<path to cuda install> NCCL_HOME=<path to nccl install> MPI_HOME=<path to mpi install>
```
Run with MPI on 8 processes, each process use one GPU:
```shell
mpirun -np 8 ./build/bin/dgc_test 1048576 1024
```
