export CUDA_VISIBLE_DEVICES=6,7

python -m paddle.distributed.launch \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/distill_gpt_345M_single_card.yaml
