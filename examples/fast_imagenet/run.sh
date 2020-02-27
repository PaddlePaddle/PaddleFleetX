export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.96
export FLAGS_eager_delete_tensor_gb=0.0
export GLOG_logtostderr=1
export NCCL_DEBUG=INFO
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_fuse_parameter_memory_size=16
export FLAGS_fuse_parameter_groups_size=50
#export GLOG_v=10
export PATH=~/sandyhouse/python/bin:$PATH

LOGDIR="mylog"
python -m paddle.distributed.launch \
  --selected_gpus="0,1,2,3,4,5,6,7" \
  --log_dir=${LOGDIR} \
  train.py --data_dir=./data \
  --use_fp16=True \
  --scale_loss=128.0 \
  --data_layout="NHWC" \
  --num_epochs=30 \
  --start_test_pass=0 --log_period=100 
