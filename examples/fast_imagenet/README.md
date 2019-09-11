# Reproducing the issue of fuse on multiple machines

## Dataset
Using the following command to get the dataset:

hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/python_1.5.tar.gz ./

## Flags used on PaddleCloud
```
export FLAGS_fraction_of_gpu_memory_to_use=0.34
export FLAGS_eager_delete_tensor_gb=0.0
export GLOG_logtostderr=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export FLAGS_sync_nccl_allreduce=0
export FLAGS_cudnn_exhaustive_search=0
export FLAGS_fuse_parameter_memory_size=16 #MB
export FLAGS_fuse_parameter_groups_size=50
```

## How to run:
```
python launch.py train.py --fp16=False --data_dir=../fast_resnet_data --nccl_comm_num=3 \
    --use_hierarchical_allreduce=True --fuse=False --use_hierarchical_allreduce=True \
    --hierarchical_allreduce_inter_nranks=8
```
