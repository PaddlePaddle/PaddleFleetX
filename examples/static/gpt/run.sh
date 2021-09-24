gpu_card=$1
CUDA_VISIBLE_DEVICES=${gpu_card}
output_dir=$2
mkdir "./output/"
mkdir $output_dir
rm -rf $output_dir/*
rm -rf ./data/*.npy

python3 -m paddle.distributed.fleet.launch \
    --log_dir ${output_dir} \
    --gpus=${gpu_card} \
    test_hybrid_parallel.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --input_dir "./data" \
    --output_dir ${output_dir} \
    --max_seq_len 512 \
    --use_amp false \
    --use_recompute false \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 500000 \
    --save_steps 10000000 \
    --decay_steps 32000000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 0 \
    --logging_freq 1\
    --eval_freq 100000 \
    --device "gpu" \
    ${@:3}
