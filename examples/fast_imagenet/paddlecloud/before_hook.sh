#!/usr/bin/env bash
echo "==============JOB BEGIN============"

echo "Network Interfaces:"
ifconfig

echo "GPU Info:"
nvidia-smi

echo "fetch gcc..."
mkdir /opt/compiler
tar zxf ./afs/gcc-4.8.2.tar.gz
mv gcc-4.8.2 /opt/compiler

echo "fetch python..."
tar zxf ./afs/python_latest.tar.gz

echo "fetch data..."
tar zxf ./afs/fast_resnet_data.tar.gz
cp ./afs/sorted_idxar.p ./fast_resnet_data/

echo "fetch CUDA environment"
tar zxf ./afs/cuda-9.2.tar.gz
tar zxf ./afs/cudnn742c92.tgz
tar zxf ./afs/nccl2.3.7_cuda9.2.tar.gz

export LD_LIBRARY_PATH=`pwd`/cuda-9.2/lib64:`pwd`/cudnn742c92/lib64:`pwd`/nccl2.3.7_cuda9.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/cuda-9.2/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

mkdir utils
mv fp16_utils.py learning_rate.py utils
mv __init__.py utils

export NCCL_DEBUG=INFO
export FLAGS_fraction_of_gpu_memory_to_use=0.96
export FLAGS_sync_nccl_allreduce=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_cudnn_exhaustive_search=1
#export FLAGS_fuse_parameter_memory_size=16
export FLAGS_fuse_parameter_memory_size=30
export FLAGS_fuse_parameter_groups_size=50

export PATH=`pwd`/python/bin:$PATH
export PYTHONPATH=`pwd`/python/lib/python2.7/site-packages:$PYTHONPATH

if [ $PADDLE_TRAINERS ];then
   config="--cluster_node_ips=${PADDLE_TRAINERS} --node_ip=${POD_IP} "
else
    config=" "
fi

FP16=True
if [ "${FP16}" = "True" ]; then
    SCALE_LOSS=64.0
else
    SCALE_LOSS=1.0
fi
LOGDIR="mylog_fp${FP16}"
python -m paddle.distributed.launch ${config} \
  --selected_gpus="0,1,2,3,4,5,6,7" \
  --log_dir=${LOGDIR} \
  train.py --data_dir=./fast_resnet_data/ \
  --num_epochs=28 \
  --fp16=${FP16} \
  --scale_loss=${SCALE_LOSS} \
  --start_test_pass=26 \
  --log_period=100 \
  --fuse=True \
  --profile=False 
  
FP16=False
if [ "${FP16}" = "True" ]; then
    SCALE_LOSS=64.0
else
    SCALE_LOSS=1.0
fi
LOGDIR="mylog_fp${FP16}"
python -m paddle.distributed.launch ${config} \
  --selected_gpus="0,1,2,3,4,5,6,7" \
  --log_dir=${LOGDIR} \
  train.py --data_dir=./fast_resnet_data/ \
  --num_epochs=30 \
  --fp16=${FP16} \
  --scale_loss=${SCALE_LOSS} \
  --start_test_pass=26 \
  --log_period=100 \
  --fuse=True \
  --profile=False 
 
