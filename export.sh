export CUDA_VISIBLE_DEVICES=7

python3.7 tools/export.py \
    -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
    -o Engine.save_load.ckpt_dir=./output/epoch_0_step_4000/
