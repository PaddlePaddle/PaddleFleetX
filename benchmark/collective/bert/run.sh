
#!/bin/bash

unset http_proxy
unset https_proxy
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

SAVE_STEPS=10000
BATCH_SIZE=2048
LR_RATE=1e-4
WEIGHT_DECAY=0.01
IN_TOKENS=true
MAX_LEN=128
MAX_PREDS_PER_SEQ=20
#TRAIN_DATA_DIR=data/tf_record_small
TRAIN_DATA_DIR=data/train
VALIDATION_DATA_DIR=data/validation
CONFIG_PATH=/home/users/dongdaxiang/github_develop/bert/uncased_L-24_H-1024_A-16/bert_config.json
VOCAB_PATH=/home/users/dongdaxiang/github_develop/bert/uncased_L-24_H-1024_A-16/vocab.txt
export FLAGS_eager_delete_tensor_gb=0
is_distributed=true

python -m paddle.distributed.launch --gpus 8 \
       train_with_executor.py --is_distributed ${is_distributed}\
       --use_cuda true\
       --weight_sharing true\
       --batch_size ${BATCH_SIZE} \
       --data_dir ${TRAIN_DATA_DIR} \
       --use_tfrecord false \
       --validation_set_dir ${VALIDATION_DATA_DIR} \
       --bert_config_path ${CONFIG_PATH} \
       --vocab_path ${VOCAB_PATH} \
       --generate_neg_sample true\
       --checkpoints ./output \
       --save_steps ${SAVE_STEPS} \
       --learning_rate ${LR_RATE} \
       --weight_decay ${WEIGHT_DECAY:-0} \
       --max_seq_len ${MAX_LEN} \
       --max_preds_per_seq ${MAX_PREDS_PER_SEQ} \
       --skip_steps 20 \
       --validation_steps 1000 \
       --num_iteration_per_drop_scope 10 \
       --use_fp16 false \
       --use_dynamic_loss_scaling false \
       --in_tokens ${IN_TOKENS} \
       --init_loss_scaling 8.0

