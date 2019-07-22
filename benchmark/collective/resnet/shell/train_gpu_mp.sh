#!/bin/bash

set -xe

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.1

# pretrain config
SAVE_STEPS=10000
BATCH_SIZE=32
LR_RATE=1e-4
WEIGHT_DECAY=0.01
MAX_LEN=128

LOG_DIR=./logs_gpu_mp_`date +%Y%m%d%H%M%S`_trainer_${PADDLE_TRAINER_ID:-0}

# Change your training arguments:
python -u ./launch_local.py --gpus ${CUDA_VISIBLE_DEVICES} --log_dir ${LOG_DIR} ./dist_train.py \
       --data_dir /ssd2/lilong/ImageNet
