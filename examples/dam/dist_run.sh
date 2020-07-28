export CUDA_VISIBLE_DEVICES=0,1,2,3

# suggested configuration
export FLAGS_fuse_parameter_memory_size=16
export FLAGS_fuse_parameter_groups_size=50
export FLAGS_conv_workspace_size_limit=4000
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

# ubuntu
python -m paddle.distributed.launch \
    --selected_gpus=0,1,2,3 \
    --log_dir=logs \
    train.py --distributed \
             --do_train True \
             --filelist train.ubuntu.files \
             --save_path ./model_files/ubuntu \
             --vocab_size 434512 \
             --batch_size 256 \
             --num_scan_data 1 \
             --data_source ubuntu \
             --vocab_path data/ubuntu/word2id

"""
# douban
python -m paddle.distributed.launch \
    --selected_gpus=0,1,2,3 \
    --log_dir=logs \
    train.py --distributed \
            --do_train True \
            --filelist train.douban.files \
            --save_path ./model_files/douban \
            --vocab_size 172130 \
            --vocab_path data/douban/word2id \
            --data_source douban \
            --channel1_num 16 \
             --num_scan_data 1 \
            --batch_size 32
"""
