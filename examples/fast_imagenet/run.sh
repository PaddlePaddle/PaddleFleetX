export PATH=/home/lilong/padd_dev/python/bin:$PATH
export FLAGS_sync_nccl_allreduce=0
export FLAGS_fraction_of_gpu_memory_to_use=0.34
export FLAGS_eager_delete_tensor_gb=0.0
#export GLOG_v=10
export GLOG_logtostderr=1
export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1
export FLAGS_cudnn_exhaustive_search=0
export FLAGS_fuse_parameter_memory_size=16 #MB
export FLAGS_fuse_parameter_groups_size=50

FP16=True
LR=2.0
PROFILE=False
SCALE_LOSS=128.0
BS_DECAY=1.0
LOGDIR="mylog_fp${FP16}_p${PROFILE}_bs${BS_DECAY}_256_224_128"
python -m paddle.distributed.launch ${config} \
  --selected_gpus="0,1,2,3,4,5,6,7" \
  --log_dir=${LOGDIR} \
  train.py --data_dir=/ssd2/lilong/fast_resnet_data/ \
  --bs_decay=${BS_DECAY} \
  --bs0=256 \
  --bs1=224 \
  --bs2=128 \
  --num_epochs=28 --lr=${LR} --fp16=${FP16} \
  --scale_loss=${SCALE_LOSS} \
  --start_test_pass=0 --log_period=100 --nccl_comm_num=2 \
  --hierarchical_allreduce_inter_nranks=8 \
  --use_hierarchical_allreduce=False \
  --enable_backward_op_deps=True \
  --fuse=True \
  --profile=False \
  --start_profile_batch=10 \
  --stop_profile_batch=16
