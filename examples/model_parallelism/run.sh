export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch mp_dygraph.py
