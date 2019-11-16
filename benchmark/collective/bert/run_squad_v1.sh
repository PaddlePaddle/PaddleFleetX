#!/bin/bash
set -xe

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# customize your own path
BERT_BASE_PATH="uncased_L-12_H-768_A-12"
CHECKPOINT_PATH=./checkpoints/
SQUAD_PATH=/home/data/bert/squad_data/

python -u run_squad.py --use_cuda true\
                   --batch_size 12 \
                   --in_tokens false\
                   --init_pretraining_params ${BERT_BASE_PATH}/params \
                   --checkpoints ${CHECKPOINT_PATH} \
                   --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                   --do_train true \
                   --do_predict true \
                   --save_steps 100 \
                   --warmup_proportion 0.1 \
                   --weight_decay  0.01 \
                   --epoch 2 \
                   --max_seq_len 384 \
                   --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
                   --predict_file ${SQUAD_PATH}/dev-v1.1.json \
                   --do_lower_case true \
                   --doc_stride 128 \
                   --train_file ${SQUAD_PATH}/train-v1.1.json \
                   --learning_rate 3e-5 \
                   --lr_scheduler linear_warmup_decay \
                   --skip_steps 10 \

