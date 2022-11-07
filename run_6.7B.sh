export CUDA_VISIBLE_DEVICES=0,1,2,3

python3.7 -m paddle.distributed.launch --log_dir logs --devices "0,1,2,3" \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_sharding8_prune_quant.yaml
