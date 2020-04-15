#!/bin/bash
set -xe
export FLAGS_sync_nccl_allreduce=1
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_batchnorm_spatial_persistent=1

export GLOG_v=1
export GLOG_logtostderr=1
export GLOG_vmodule=gen_nccl_id_op=10

export FLAGS_eager_delete_tensor_gb=0
export NCCL_DEBUG=INFO
# Unset proxy
unset https_proxy http_proxy


set -xe

if [[ ${FUSE} == "True" ]]; then
    export FLAGS_fuse_parameter_memory_size=16 #MB
    export FLAGS_fuse_parameter_groups_size=50
fi

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PYTHONPATH=$SCRIPTPATH/../:$PYTHONPATH
echo $PYTHONPATH

DATA_PATH="../ImageNet"

#export PADDLE_TRAINERS=2
#export PADDLE_TRAINERS_NUM=2
#export PADDLE_TRAINER_ID=0 
#export FLAGS_selected_gpus=4
#python ./remote_executor.py  --data_dir=$DATA_PATH > 0.log 2>&1 &


#export PADDLE_TRAINER_ID=1
#export FLAGS_selected_gpus=5
#python ./remote_executor.py --data_dir=$DATA_PATH > 1.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4,5

python -m paddle.distributed.launch  --log_dir log \
    remote_executor.py --data_dir $DATA_PATH
