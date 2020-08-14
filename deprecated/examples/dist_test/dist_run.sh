export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# suggested configuration
export FLAGS_fuse_parameter_memory_size=16
export FLAGS_fuse_parameter_groups_size=50
export FLAGS_conv_workspace_size_limit=4000
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

python -m paddle.distributed.launch \
    --selected_gpus=0,1 \
    --log_dir=mylog \
    train.py --distributed
