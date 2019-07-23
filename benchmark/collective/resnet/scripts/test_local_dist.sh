#!/bin/bash
set -xe

# Paddle debug envs
export GLOG_v=1
export GLOG_logtostderr=1

# Unset proxy
unset https_proxy http_proxy

# NCCL debug envs
export NCCL_DEBUG=INFO
# Comment it if your nccl support IB
export NCCL_IB_DISABLE=1

# Add your nodes endpoints here.
export PADDLE_TRAINERS=127.0.0.1,127.0.0.1
export POD_IP=127.0.0.1
export PADDLE_TRAINER_ID=0
export CUDA_VISIBLE_DEVICES=0,1
bash shell/train_gpu_mp.sh > 0.log 2>&1 &

# Add your nodes endpoints here.
export PADDLE_TRAINERS=127.0.0.1,127.0.0.1
export POD_IP=127.0.0.1
export PADDLE_TRAINER_ID=1
export CUDA_VISIBLE_DEVICES=4,5
export PATH=../python/bin:$PATH
bash shell/train_gpu_mp.sh > 1.log 2>&1 &
