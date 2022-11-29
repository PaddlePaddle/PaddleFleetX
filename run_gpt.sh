
#export LD_PRELOAD=/usr/local/lib/python3.7/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0 # for npu

export PYTHONPATH=/Paddle/Paddle/build2/python:/Paddle/fleetx/PaddleSlim:$PYTHONPATH
export BKCL_PCIE_RING=1

export XPU_VISIBLE_DEVICES=4,5,6,7 #比如x=0
rm -rf log_gpt
python -m paddle.distributed.launch \
--devices ${XPU_VISIBLE_DEVICES} --log_dir log_gpt \
tools/train.py \
-c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
-o Engine.mix_precision.use_pure_fp16="False" \
-o Distributed.dp_degree=2 \
-o Distributed.mp_degree=1 \
-o Distributed.pp_degree=1 \
-o Distributed.sharding.sharding_degree=2 \
-o Distributed.sharding.sharding_stage=2 \
-o Model.use_recompute="False" \
-o Global.local_batch_size=2 \
-o Global.micro_batch_size=2 \
-o Global.device=xpu \
-o Model.num_layers=4 \
-o Engine.max_steps=10
