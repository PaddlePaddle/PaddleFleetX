#!/bin/bash
set -xe

if [ $PADDLE_TRAINERS ]; then
    echo "PADDLE_TRAINERS = $PADDLE_TRAINERS"
else
    PADDLE_TRAINERS="127.0.0.1"
    echo "PADDLE_TRAINERS = $PADDLE_TRAINERS"
fi

if [ $POD_IP ]; then
    echo "POD_IP = $POD_IP"
else
    POD_IP="127.0.0.1"
    echo "POD_IP = $POD_IP"
fi

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.84
export NCCL_DEBUG=INFO

# Change your training arguments:
python -m paddle.distributed.launch --cluster_node_ips=$PADDLE_TRAINERS \
    --node_ip=$POD_IP ./train_with_pe_fleet.py \
    --data_dir /ssd2/lilong/ImageNet --start_test_pass 0
