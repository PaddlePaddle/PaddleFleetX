CUDA_VISIBLE_DEVICES="0"

output_dir="./output/serial"
mkdir "./output/"
mkdir $output_dir
rm -rf $output_dir/*
rm -rf ./data/*.npy

python3 -m paddle.distributed.fleet.launch \
    --log_dir ${output_dir} \
    --gpus="0" \
    test_auto_parallel.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --input_dir "./data" \
    --output_dir ${output_dir} \
    --max_seq_len 512 \
    --use_amp false \
    --use_recompute false \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 30 \
    --save_steps 10 \
    --decay_steps 32000000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 0 \
    --logging_freq 1\
    --eval_freq 100000 \
    --device "gpu" \
    --global_batch_size 4 \
    --mp_degree 1 \
    --dp_degree 1 \
    --pp_degree 1
