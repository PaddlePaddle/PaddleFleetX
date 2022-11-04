# export LD_PRELOAD=/usr/local/lib/python3.7/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
export PYTHONPATH=/code_lp/paddle/Paddle/build/python
python tools/train.py \
-c ppfleetx/configs/nlp/ernie/pretrain_ernie_base_345M_single_card.yaml \
-o Data.Train.dataset.input_dir="input_dir" \
-o Data.Train.dataset.tokenizer_type=ernie-1.0-base-zh \
-o Data.Eval.dataset.input_dir="input_dir" \
-o Data.Eval.dataset.tokenizer_type=ernie-1.0-base-zh \
-o Engine.mix_precision.use_pure_fp16="True"