
# for mp=8(GPT 175b)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" projects/gpt/benchmark.py --seq_len 128 --iter 10 --mp_size 8 --model_dir ./output

# for mp=1(GPT 6.7B & GPT 345M)
export CUDA_VISIBLE_DEVICES=0
python -m -m paddle.distributed.launch --devices "0" projects/gpt/benchmark.py --seq_len 128 --iter 10 --mp_size 1 --model_dir ./output