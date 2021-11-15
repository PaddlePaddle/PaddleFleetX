CUDA_VISIBLE_DEVICES="0,1,2,3"

output_dir="./output/autosearch"
mkdir "./output/"
mkdir $output_dir
rm -rf $output_dir/*
rm -rf ./data/*.npy

python3 -m paddle.distributed.fleet.launch \
    --log_dir ${output_dir} \
    --gpus="0,1,2,3" \
    test_auto_parallel_gpt.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --input_dir "./data" \
    --output_dir ${output_dir} \
    --max_seq_len 512 \
    --use_amp false \
    --use_recompute false \
    --auto_search true \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 100 \
    --save_steps 10 \
    --decay_steps 32000000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 0 \
    --logging_freq 1\
    --eval_freq 100000 \
    --device "gpu" \
    --global_batch_size 8 \
    --mp_degree 1 \
    --dp_degree 1 \
    --pp_degree 1 
