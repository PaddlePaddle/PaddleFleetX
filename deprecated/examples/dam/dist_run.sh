export CUDA_VISIBLE_DEVICES=0,1,2,3

# suggested configuration
export FLAGS_fuse_parameter_memory_size=16
export FLAGS_fuse_parameter_groups_size=50
export FLAGS_conv_workspace_size_limit=4000
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

# ubuntu dist train
python -m paddle.distributed.launch \
    --selected_gpus=0,1,2,3 \
    --log_dir=logs \
    train.py --do_train True \
             --train_data_path data/ubuntu/train.txt \
             --valid_data_path data/ubuntu/valid.txt \
             --word_emb_init data/ubuntu/word_embedding.pkl \
             --vocab_path data/ubuntu/word2id \
             --data_source ubuntu \
             --save_path ./model_files/ubuntu \
             --vocab_size 434512 \
             --batch_size 64 \
             --num_scan_data 2

# douban dist train
python -m paddle.distributed.launch \
    --selected_gpus=0,1,2,3 \
    --log_dir=logs \
    train.py --do_train True \
             --train_data_path data/douban/train.txt \
             --valid_data_path data/douban/dev.txt \
             --word_emb_init data/douban/word_embedding.pkl \
             --vocab_path data/douban/word2id \
             --data_source douban \
             --save_path ./model_files/douban \
             --vocab_size 172130 \
             --channel1_num 16 \
             --batch_size 64 \
             --num_scan_data 2
