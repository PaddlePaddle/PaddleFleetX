#!/bin/bash
export FLAGS_cudnn_exhaustive_search=0

export GLOG_v=1
export GLOG_logtostderr=1
export FLAGS_eager_delete_tensor_gb=0
export NCCL_DEBUG=INFO
# Unset proxy
unset https_proxy http_proxy

set -xe

MODEL=ResNet50 #VGG16
MODEL_SAVE_PATH="output/"

# training params
NUM_EPOCHS=90
BATCH_SIZE=32
LR=0.001
LR_STRATEGY=piecewise_decay

# data params
DATA_PATH="./ImageNet"
TOTAL_IMAGES=1281167
CLASS_DIM=1000
IMAGE_SHAPE=3,224,224


#gpu params
FUSE=True
NCCL_COMM_NUM=2
NUM_CARDS=4
FP16=False #whether to use float16 

distributed_args="--selected_gpus `seq -s, 0 $(($NUM_CARDS-1))`"

set -x

python -m paddle.distributed.launch ${distributed_args} --log_dir log \
       ./train_with_fleet.py \
       --model=${MODEL} \
       --batch_size=${BATCH_SIZE} \
       --total_images=${TOTAL_IMAGES} \
       --data_dir=${DATA_PATH} \
       --class_dim=${CLASS_DIM} \
       --image_shape=${IMAGE_SHAPE} \
       --model_save_dir=${MODEL_SAVE_PATH} \
       --lr_strategy=${LR_STRATEGY} \
       --lr=${LR} \
       --num_epochs=${NUM_EPOCHS} \
       --l2_decay=1e-4 \
       --scale_loss=1.0 \
       --nccl_comm_num=${NCCL_COMM_NUM} \
       --fp16=${FP16} \
       --use_local_sgd=True \
       --local_sgd_steps=2
