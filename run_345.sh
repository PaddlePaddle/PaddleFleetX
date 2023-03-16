#export LD_PRELOAD=/usr/local/lib/python3.7/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
#export PYTHONPATH=/workspace/paddle/Paddle/built-in/python:$PYTHONPATH
#export PADDLE_XCCL_BACKEND=hccl
#export HCCL_VISIBLE_DEVICES=0,1
#export FLAGS_enable_eager_mode=1
python -c "import paddle; print(paddle.version.commit)"
python -m paddle.distributed.launch \
--devices 2,3 --log_dir log_ernie \
tools/train.py \
-c ppfleetx/configs/nlp/ernie/pretrain_ernie_base_345M_single_card.yaml \
-o Data.Train.dataset.input_dir="input_dir" \
-o Data.Train.dataset.tokenizer_type=ernie-1.0-base-zh \
-o Data.Eval.dataset.input_dir="input_dir" \
-o Data.Eval.dataset.tokenizer_type=ernie-1.0-base-zh \
-o Engine.mix_precision.use_pure_fp16="False" \
-o Model.num_attention_heads=16 \
-o Model.num_hidden_layers=24 \
-o Distributed.dp_degree=2 \
-o Global.device=npu \
-o Global.local_batch_size=8 \
-o Global.micro_batch_size=8

#Model:
#   vocab_size: 40000
#   hidden_size: 1024
#   num_hidden_layers: 24
#   num_attention_heads: 16

# Distributed:
#   dp_degree: 1
#   mp_degree: 1
#   pp_degree: 1
# Global:
#   device: gpu
