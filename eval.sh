export CUDA_VISIBLE_DEVICES=3

# python3.7 ./tools/eval.py -c ./ppfleetx/configs/nlp/gpt/eval_gpt_6.7B_single_card_prune.yaml
python3.7 ./tools/eval.py -c ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card_prune_quant.yaml

