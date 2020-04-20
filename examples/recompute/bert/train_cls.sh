export FLAGS_enable_parallel_graph=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

export FLAGS_fraction_of_gpu_memory_to_use=0.99
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fuse_parameter_memory_size=32 #MB
export FLAGS_fuse_parameter_groups_size=50
export FLAGS_allocator_strategy=naive_best_fit

PRETRAINED_CKPT_PATH=$1
TASK_NAME='XNLI'
CKPT_PATH=$PWD/tmp
bert_config_path=$2
vocab_path=$3
DATA_PATH=$4
BATCH_SIZE=16

python -m paddle.distributed.launch --log_dir mylog \
           run_classifier.py --task_name ${TASK_NAME} \
                   --use_cuda true \
                   --do_train true \
                   --do_val true \
                   --do_test false \
                   --batch_size ${BATCH_SIZE} \
                   --in_tokens false \
                   --init_pretraining_params ${PRETRAINED_CKPT_PATH} \
                   --data_dir ${DATA_PATH} \
                   --vocab_path ${vocab_path} \
                   --checkpoints ${CKPT_PATH} \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 50 \
                   --epoch 3 \
                   --max_seq_len 512 \
                   --bert_config_path ${bert_config_path} \
                   --learning_rate 5e-5 \
                   --skip_steps 1 \
                   --use_recompute true \
                   --use_mix_precision false
