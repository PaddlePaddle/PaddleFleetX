DATA_DIR=./gen_data

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m paddle.distributed.launch \
    --cluster_node_ips=127.0.0.1 --node_ip=127.0.0.1 \
    --selected_gpus="0,1,2,3,4,5,6,7" --log_dir=mylog \
    train.py \
    --src_vocab_fpath ${DATA_DIR}/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
    --trg_vocab_fpath ${DATA_DIR}/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
    --special_token '<s>' '<e>' '<unk>' \
    --train_file_pattern ${DATA_DIR}/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
    --val_file_pattern ${DATA_DIR}/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
    --token_delimiter ' ' \
    --use_token_batch True \
    --batch_size 3200 \
    --sort_type pool \
    --num_epochs 30 \
    --pool_size 200000 \
    n_head 16 \
    d_model 1024 \
    d_inner_hid 4096 \
    prepostprocess_dropout 0.3
