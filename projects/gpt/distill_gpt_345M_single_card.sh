export CUDA_VISIBLE_DEVICES=0,1

python -m paddle.distributed.launch \
    --log_dir distill_logs \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/distill_gpt_345M_single_card.yaml \
