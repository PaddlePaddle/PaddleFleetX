# export PYTHONPATH=/gpt3/Paddle/build/python
# export PYTHONPATH=/gpt3/PaddleFleetX/paddle_test
export PYTHONPATH=/code_lp/paddle/Paddle/build/python
# unset PYTHONPATH
rm -rf $log
log=output_345
python -m paddle.distributed.launch --log_dir $log --devices "1" \
    tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Engine.max_steps=2 \
    -o Engine.save_load.output_dir="$log" \
    -o Engine.save_load.ckpt_dir="pretrained/PaddleFleetX_GPT_345M_220826/"
