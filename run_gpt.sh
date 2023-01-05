
set -uex
get_data(){
    set -e
    mkdir -p data
    # DATA_HOST and $DATA_PORT are not open source
    wget ${DATA_HOST}:${DATA_PORT}/$1 -O data/$1
    set +e
}

prepare_data(){

  [[ `md5sum data/gpt_openwebtext_0827_ids.npy|cut -f 1 -d ' '` = b2286ed5807368f7a74a661eba7f4556 ]] || get_data gpt_openwebtext_0827_ids.npy
  [[ `md5sum data/gpt_openwebtext_0827_idx.npz|cut -f 1 -d ' '` = 8ec84951340abf5732a62255573a408f ]] || get_data gpt_openwebtext_0827_idx.npz

  find data -type f |grep -v gpt_openwebtext_0827_ids.npy|grep -v gpt_openwebtext_0827_idx.npz|xargs rm -f
}


set_gpu_config(){
    gpu=gpu

}

set_dcu_config(){
    gpu=gpu
}

set_xpu_config(){
    gpu=xpu
    export PYTHONPATH=/Paddle/Paddle/build/python/:/Paddle/fleetx/PaddleSlim:$PYTHONPATH
    export BKCL_PCIE_RING=1
}

set_npu_config(){
    gpu=npu
}

set_xpu_config
npu-smi info && set_npu_config || true
nvidia-smi && set_gpu_config || true
rocm-smi && set_dcu_config || true
device=$gpu


train_345M(){
  # 345M
  # fp16 4 卡GPU 66天训完 speed: 0.10 step/s
  # fp32 4 卡GPU 198天训完 speed: 0.03 steps/s, gpu 8卡 0.05 steps/s, xpu 4卡 0.03 steps/s
  # micro_batch_size: gpu 设置32, xpu设置 16
  log_dir=log_345M
  dp_degree=8
  python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
      tools/train.py \
      -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Global.local_batch_size=$((512/dp_degree)) \
    -o Global.micro_batch_size=16 \
    -o Optimizer.weight_decay=0.1 \
    -o Optimizer.beta1=0.9 \
    -o Optimizer.beta2=0.95 \
    -o Opitmizer.lr.max_lr=3.0e-4 \
    -o Optimizer.lr.min_lr=1.0e-5 \
    -o Engine.max_steps=572000 \
    -o Model.use_recompute=True \
    -o Engine.mix_precision.use_pure_fp16=True \
    -o Distributed.dp_degree=$dp_degree \
    -o Global.device=$device
}


train_345M_pp(){
  # 345M
  # fp16 4 卡GPU 66天训完 speed: 0.10 step/s
  # fp32 4 卡GPU 198天训完 speed: 0.03 steps/s, gpu 8卡 0.05 steps/s, xpu 4卡 0.03 steps/s
  # micro_batch_size: gpu 设置32, xpu设置 16
  log_dir=log_345M_pp
  dp_degree=1
  mp_degree=1
  pp_degree=2
  python -u -m paddle.distributed.launch --log_dir $log_dir --devices "0,1" \
      tools/train.py \
      -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Global.local_batch_size=16 \
    -o Global.micro_batch_size=2 \
    -o Optimizer.weight_decay=0.1 \
    -o Optimizer.beta1=0.9 \
    -o Optimizer.beta2=0.95 \
    -o Opitmizer.lr.max_lr=3.0e-4 \
    -o Optimizer.lr.min_lr=1.0e-5 \
    -o Engine.max_steps=10 \
    -o Model.use_recompute=False \
    -o Engine.mix_precision.use_pure_fp16=False \
    -o Distributed.dp_degree=$dp_degree \
    -o Distributed.mp_degree=$mp_degree \
    -o Distributed.pp_degree=$pp_degree \
    -o Global.device=$device
}


train_345M_mp(){
  # 345M
  # fp16 4 卡GPU 66天训完 speed: 0.10 step/s
  # fp32 4 卡GPU 198天训完 speed: 0.03 steps/s, gpu 8卡 0.05 steps/s, xpu 4卡 0.03 steps/s
  # micro_batch_size: gpu 设置32, xpu设置 16
  log_dir=log_345M_mp
  dp_degree=1
  mp_degree=2
  pp_degree=1
  python -u -m paddle.distributed.launch --log_dir $log_dir --devices "0,1" \
      tools/train.py \
      -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Global.local_batch_size=16 \
    -o Global.micro_batch_size=2 \
    -o Optimizer.weight_decay=0.1 \
    -o Optimizer.beta1=0.9 \
    -o Optimizer.beta2=0.95 \
    -o Opitmizer.lr.max_lr=3.0e-4 \
    -o Optimizer.lr.min_lr=1.0e-5 \
    -o Engine.max_steps=10 \
    -o Model.use_recompute=False \
    -o Engine.mix_precision.use_pure_fp16=False \
    -o Distributed.dp_degree=$dp_degree \
    -o Distributed.mp_degree=$mp_degree \
    -o Distributed.pp_degree=$pp_degree \
    -o Global.device=$device
}

train_345M_sharding(){
  # 345M
  # fp16 4 卡GPU 66天训完 speed: 0.10 step/s
  # fp32 4 卡GPU 198天训完 speed: 0.03 steps/s, xpu 4卡 0.01 step/s 
  # micro_batch_size: gpu : 32, xpu: 16 
  log_dir=log_345M_sharding
  sharding_degree=4
  python -m paddle.distributed.launch --log_dir $log_dir --devices "4,5,6,7" \
      tools/train.py \
      -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Global.local_batch_size=$((512/sharding_degree)) \
    -o Global.micro_batch_size=16 \
    -o Optimizer.weight_decay=0.1 \
    -o Optimizer.beta1=0.9 \
    -o Optimizer.beta2=0.95 \
    -o Opitmizer.lr.max_lr=3.0e-4 \
    -o Optimizer.lr.min_lr=1.0e-5 \
    -o Engine.max_steps=572000 \
    -o Model.use_recompute=True \
    -o Engine.mix_precision.use_pure_fp16=False \
    -o Distributed.sharding.sharding_degree=${sharding_degree} \
    -o Distributed.sharding.sharding_stage=2 \
    -o Distributed.dp_degree=1 \
    -o Global.device=$device
}

train_1_3B(){
  # 1.3B fp16 GPU 4卡 0.02step/s 训完需要173天
  log_dir=log_1.3B
  dp_degree=4
  python -m paddle.distributed.launch --log_dir $log_dir --devices "4,5,6,7" \
      tools/train.py \
      -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml \
    -o Optimizer.weight_decay=0.1 \
    -o Optimizer.beta1=0.9 \
    -o Optimizer.beta2=0.95 \
    -o Opitmizer.lr.max_lr=0.0002 \
    -o Optimizer.lr.min_lr=1.0e-5 \
    -o Engine.max_steps=300000 \
    -o Model.use_recompute=True \
    -o Engine.mix_precision.use_pure_fp16=True \
    -o Distributed.dp_degree=$dp_degree \
    -o Global.local_batch_size=$((1024/dp_degree)) \
    -o Global.micro_batch_size=4 \
    -o Global.device=$device

}

train6.7B(){
  # 6.7B fp16 sharding_degree=16 无最佳配置 最少2机
  log_dir=log_6.7B
  python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
      tools/train.py \
      -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
      -o Global.device=$device
}

# 可以只执行一次， 数据下载下来之后可以注释掉
# prepare_data

train_345M_mp
# train_345M_sharding
# train_1_3B
