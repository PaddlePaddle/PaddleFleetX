export PATH=~/paddle_distfc/python/bin:$PATH
if [ $PADDLE_TRAINERS ];then
   config="--cluster_node_ips=${PADDLE_TRAINERS} --node_ip=${POD_IP} "
else
    config=" "
fi
selected_gpus="0,1,2,3,4,5,6,7"

export FLAGS_cudnn_exhaustive_search=true 
export FLAGS_fraction_of_gpu_memory_to_use=0.96
export FLAGS_eager_delete_tensor_gb=0.0

python -m paddle.distributed.launch $config \
  --selected_gpus $selected_gpus \
  --log_dir mylog_distsoftmax \
  do_train.py \
  --model=ResNet_ARCFACE50 \
  --loss=softmax \
  --margin=0.5 \
  --train_batch_size 128 \
  --with_test=False
