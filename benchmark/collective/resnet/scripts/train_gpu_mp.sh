#!/bin/bash

export PATH=~/software/python/bin:$PATH
set -xe

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.84
export NCCL_DEBUG=INFO

# Change your training arguments:
python -u ./launch.py --gpus 8 ./train_with_pe_fleet.py \
       --data_dir /ssd2/lilong/ImageNet --start_test_pass 0
