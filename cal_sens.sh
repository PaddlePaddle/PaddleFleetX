export CUDA_VISIBLE_DEVICES=4
  
python3.7 ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card_prune_sens.yaml
