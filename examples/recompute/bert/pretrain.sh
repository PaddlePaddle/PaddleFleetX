#!/bin/bash
set -xe

export FLAGS_enable_parallel_graph=0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fuse_parameter_memory_size=32 #MB
export FLAGS_fuse_parameter_groups_size=50
export FLAGS_allocator_strategy=naive_best_fit

cluster_node_ips="127.0.0.1"
node_ip="127.0.0.1"

distributed_args=""
if [[ $# -ge 1 ]]; then
    while true ; do
        case "$1" in
        -cluster_node_ips) cluster_node_ips="$2" ; shift 2 ;;
        -node_ip) node_ip="$2" ; shift 2 ;;
        *)
           if [[ ${#1} > 0 ]]; then
              echo "not supported arugments ${1}" ; exit 1 ;
           else
               break
           fi
           ;;
        esac
    done
  distributed_args="--cluster_node_ips ${cluster_node_ips} --node_ip ${node_ip}"
fi


# pretrain config
SAVE_STEPS=10000
BATCH_SIZE=56
LR_RATE=1e-4
WEIGHT_DECAY=0.01
MAX_LEN=512
TRAIN_DATA_DIR=$PWD/../../../benchmark/collective/bert/data/train
VALIDATION_DATA_DIR=$PWD/../../../benchmark/collective/bert/data/validation
CONFIG_PATH=data/large_config/bert_config.json
VOCAB_PATH=data/large_config/vocab.txt

export CUDA_VISIBLE_DEVICES=0,1

# Change your train arguments:
rm -rf mylog
python -m paddle.distributed.launch ${distributed_args}  --log_dir mylog \
        ./train.py \
        --use_cuda true \
        --weight_sharing true \
        --batch_size ${BATCH_SIZE} \
        --data_dir ${TRAIN_DATA_DIR} \
        --validation_set_dir ${VALIDATION_DATA_DIR} \
        --bert_config_path ${CONFIG_PATH} \
        --in_tokens false \
        --vocab_path ${VOCAB_PATH} \
        --generate_neg_sample true\
        --checkpoints ./output \
        --save_steps ${SAVE_STEPS} \
        --learning_rate ${LR_RATE} \
        --weight_decay ${WEIGHT_DECAY:-0} \
        --max_seq_len ${MAX_LEN} \
        --skip_steps 1 \
        --validation_steps 10000000000 \
        --num_iteration_per_drop_scope 10 \
        --loss_scaling 8.0 \
        --profile false \
        --use_recompute true \
        --use_mix_precision false
       
