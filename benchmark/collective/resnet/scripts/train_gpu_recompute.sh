#!/bin/bash
export FLAGS_sync_nccl_allreduce=1
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_conv_workspace_size_limit=0 #MB
export FLAGS_cudnn_batchnorm_spatial_persistent=1

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
BATCH_SIZE=760
LR=0.1
LR_STRATEGY=piecewise_decay

# data params
DATA_PATH="./ImageNet"
TOTAL_IMAGES=1281167
CLASS_DIM=1000
IMAGE_SHAPE=3,224,224
DATA_FORMAT="NHWC"

#gpu params
FUSE=True
NCCL_COMM_NUM=1
NUM_THREADS=2
USE_HIERARCHICAL_ALLREDUCE=False
NUM_CARDS=8
FP16=False #whether to use float16
use_dali=False
if [[ ${use_dali} == "True" ]]; then
    export FLAGS_fraction_of_gpu_memory_to_use=0.8
fi

# dgc params
USE_DGC=False # whether to use dgc
ALL_CARDS=8
START_EPOCHS=4 # start dgc after 4 epochs
# add 1 in let, let will not return 1 when set START_EPOCHS=0
let '_tmp_ans=((TOTAL_IMAGES+BATCH_SIZE*ALL_CARDS-1)/(BATCH_SIZE*ALL_CARDS))*START_EPOCHS' 1
DGC_RAMPUP_BEGIN_STEP=${_tmp_ans}

if [[ ${FUSE} == "True" ]]; then
    export FLAGS_fuse_parameter_memory_size=16 #MB
    export FLAGS_fuse_parameter_groups_size=50
fi
distributed_args=""
if [[ ${NUM_CARDS} == "1" ]]; then
    distributed_args="--selected_gpus 0"
fi

set -x

python -m paddle.distributed.launch ${distributed_args}  --log_dir log \
       ./train_with_fleet.py \
       --model=${MODEL} \
       --batch_size=${BATCH_SIZE} \
       --total_images=${TOTAL_IMAGES} \
       --data_dir=${DATA_PATH} \
       --class_dim=${CLASS_DIM} \
       --image_shape=${IMAGE_SHAPE} \
       --data_format=${DATA_FORMAT} \
       --model_save_dir=${MODEL_SAVE_PATH} \
       --with_mem_opt=False \
       --lr_strategy=${LR_STRATEGY} \
       --lr=${LR} \
       --num_epochs=${NUM_EPOCHS} \
       --l2_decay=1e-4 \
       --scale_loss=128.0 \
       --use_dynamic_loss_scaling=True \
       --fuse=${FUSE} \
       --num_threads=${NUM_THREADS} \
       --nccl_comm_num=${NCCL_COMM_NUM} \
       --use_hierarchical_allreduce=${USE_HIERARCHICAL_ALLREDUCE} \
       --fp16=${FP16} \
       --use_dali=${use_dali} \
       --use_dgc=${USE_DGC} \
       --fetch_steps=10 \
       --do_test=True \
       --profile=False \
       --rampup_begin_step=${DGC_RAMPUP_BEGIN_STEP} \
       --use_recompute=True
