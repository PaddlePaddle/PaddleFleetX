export CUDA_VISIBLE_DEVICES=0,1,2,3

# suggested configuration
export FLAGS_fuse_parameter_memory_size=16
export FLAGS_fuse_parameter_groups_size=50
export FLAGS_conv_workspace_size_limit=4000
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

python -m paddle.distributed.launch \
    --selected_gpus=0,1,2,3 \
    --log_dir=logs \
    train.py --distributed \
             --do_train True \
             --data_path ./data/data_small.pkl \
             --save_path ./model_files/ubuntu \
             --vocab_size 434512 \
             --_EOS_ 28270 \
             --batch_size 16
