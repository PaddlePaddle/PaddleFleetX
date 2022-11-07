export CUDA_VISIBLE_DEVICES=2

python3.7 ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card_prune_quant.yaml
# python3.7 ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml
