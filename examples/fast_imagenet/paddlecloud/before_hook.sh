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
tar zxvf gcc-4.8.2.tar.gz > /dev/null
mv gcc-4.8.2 /opt/compiler
rm gcc-4.8.2.tar.gz

echo "fetch python..."
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/python_latest.tar.gz ./
echo "untar..."
tar zxvf python_latest.tar.gz > /dev/null
rm python_latest.tar.gz

echo "fetch data..."
#export PATH=/root/paddlejob/hadoop-client/hadoop/bin:$PATH
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.0 ./ &
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.1 ./ &
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.2 ./ &
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.3 ./ &
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.4 ./ &
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.5 ./ &
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.6 ./ &
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.7 ./ &
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.8 ./ &
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/fast_restnet_data_partition/fast_resnet_data.tar.gz.9 ./ &
wait
cat fast_resnet_data.tar.gz* > fast_resnet_data.tar.gz
rm fast_resnet_data.tar.gz.0 fast_resnet_data.tar.gz.1 fast_resnet_data.tar.gz.2 fast_resnet_data.tar.gz.3 fast_resnet_data.tar.gz.4 \
    fast_resnet_data.tar.gz.5 fast_resnet_data.tar.gz.6 fast_resnet_data.tar.gz.7 fast_resnet_data.tar.gz.8 fast_resnet_data.tar.gz.9
echo "untar..."
tar zxvf fast_resnet_data.tar.gz > /dev/null
rm fast_resnet_data.tar.gz

echo "fetch CUDA environment"
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/cuda-9.2.tar.gz ./
echo "untar..."
tar zxvf cuda-9.2.tar.gz > /dev/null
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/cudnn742c92.tgz ./
echo "untar..."
tar zxvf cudnn742c92.tgz > /dev/null
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/nccl2.3.7_cuda9.2.tar.gz ./
echo "untar..."
tar zxvf nccl2.3.7_cuda9.2.tar.gz > /dev/null
rm cuda-9.2.tar.gz cudnn742c92.tgz nccl2.3.7_cuda9.2.tar.gz

export LD_LIBRARY_PATH=`pwd`/cuda-9.2/lib64:`pwd`/cudnn742c92/lib64:`pwd`/nccl2.3.7_cuda9.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/cuda-9.2/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

#mkdir fast_resnet_data/sz
#mv fast_resnet_data/160 fast_resnet_data/sz/160
#mv fast_resnet_data/352 fast_resnet_data/sz/352

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
LR=1.0
PROFILE=False
SCALE_LOSS=128.0
LOGDIR="mylog_fp${FP16}_lr${LR}"
python -m paddle.distributed.launch ${config} \
  --selected_gpus="0,1,2,3,4,5,6,7" \
  --log_dir=${LOGDIR} \
  train.py --data_dir=./fast_resnet_data/ \
  --num_epochs=30 --lr=${LR} --fp16=${FP16} \
  --scale_loss=${SCALE_LOSS} \
  --start_test_pass=0 --log_period=100 --nccl_comm_num=2 \
  --fuse=True \
  --profile=False 

FP16=False
LR=1.0
PROFILE=False
SCALE_LOSS=1.0
LOGDIR="mylog_fp${FP16}_lr${LR}"
python -m paddle.distributed.launch ${config} \
  --selected_gpus="0,1,2,3,4,5,6,7" \
  --log_dir=${LOGDIR} \
  train.py --data_dir=./fast_resnet_data/ \
  --num_epochs=30 --lr=${LR} --fp16=${FP16} \
  --scale_loss=${SCALE_LOSS} \
  --start_test_pass=0 --log_period=100 --nccl_comm_num=2 \
  --fuse=True \
  --profile=False 
