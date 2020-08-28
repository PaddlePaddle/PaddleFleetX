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
MODEL_SAVE_PATH="./output"

# training params
NUM_EPOCHS=90
BATCH_SIZE=32
LR=0.1

# data params
# the path of the data set
DATA_PATH="./ImageNet"
TOTAL_IMAGES=1281167
CLASS_DIM=1000
IMAGE_SHAPE=3,224,224

# gpu params
NCCL_COMM_NUM=2
set -x
config="--selected_gpus=0,1,2,3,4,5,6,7 --log_dir mylog"
touch ./utils/__init__.py
python -m paddle.distributed.launch ${config} \
       ./train.py \
       --model=${MODEL} \
       --batch_size=${BATCH_SIZE} \
       --total_images=${TOTAL_IMAGES} \
       --data_dir=${DATA_PATH} \
       --class_dim=${CLASS_DIM} \
       --image_shape=${IMAGE_SHAPE} \
       --model_save_dir=${MODEL_SAVE_PATH} \
       --lr=${LR} \
       --num_epochs=${NUM_EPOCHS} \
       --l2_decay=1e-4 \
       --nccl_comm_num=2 \
