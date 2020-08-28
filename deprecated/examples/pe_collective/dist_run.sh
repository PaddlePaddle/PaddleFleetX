export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_fuse_parameter_memory_size=16
export FLAGS_fuse_parameter_groups_size=50

python -m paddle.distributed.launch \
    --selected_gpus=0,1 \
    --log_dir=mylog \
    train.py --distributed
