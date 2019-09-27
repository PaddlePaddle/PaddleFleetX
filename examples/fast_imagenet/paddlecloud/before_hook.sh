#!/usr/bin/env bash
echo "==============JOB BEGIN============"

# User configurations
HADOOP_FS_NAME=afs://xingtian.afs.baidu.com:9902
HADOOP_UGI=Paddle_Data,Paddle_gpu@2017

echo "Show network interfaces on this machine:"
ifconfig

echo "fetch gcc..."
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/gcc-4.8.2.tar.gz ./
mkdir /opt/compiler
tar zxf gcc-4.8.2.tar.gz 
mv gcc-4.8.2 /opt/compiler
rm gcc-4.8.2.tar.gz

echo "fetch python..."
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/python_latest.tar.gz ./
echo "untar..."
tar zxf python_latest.tar.gz 
rm python_latest.tar.gz

echo "fetch data..."
#export PATH=/root/paddlejob/hadoop-client/hadoop/bin:$PATH
tar zxf ./afs/fast_resnet_data.tar.gz
mv ./afs/sorted_idxar.p ./fast_resnet_data/

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
export FLAGS_fraction_of_gpu_memory_to_use=0.9
export FLAGS_sync_nccl_allreduce=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_fuse_parameter_memory_size=16 #MB
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
    SCALE_LOSS=128.0
else
    SCALE_LOSS=1.0
fi
LOGDIR="mylog_fp${FP16}"
python -m paddle.distributed.launch ${config} \
  --selected_gpus="0,1,2,3,4,5,6,7" \
  --log_dir=${LOGDIR} \
  train.py --data_dir=./fast_resnet_data/ \
  --num_epochs=40 --fp16=${FP16} \
  --scale_loss=${SCALE_LOSS} \
  --start_test_pass=0 --log_period=100 \
  --fuse=True \
  --profile=False 

