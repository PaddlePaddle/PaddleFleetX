#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
##                   k8s 类型作业演示                        ##
## 请将下面的 user_ak/user_sk 替换成自己的 ak/sk             ##
## 请将下面的 cluster_name 替换成所在组关联的k8s集群名称     ##
##                                                           ##
###############################################################
job_name=fast_imagenet_m1

# 线上正式环境
server="paddlecloud.baidu-int.com"
port=80

# 请替换成所在组下的个人 access key/secret key
user_ak="b932d8700e665df88b9c5d550cdd5d36"
user_sk="c4523ba6bd68532b938827e1894e2180"

# 作业参数
gpus_per_node="8"
k8s_gpu_type="baidu/gpu_v100"
k8s_wall_time="240:00:00"
k8s_memory="300Gi"
k8s_priority="high"
is_standalone="1"
k8s_trainers="1"

# 请替换成所在组关联的集群名称
cluster_name="v100-32-0-cluster"
# 作业版本
job_version="paddle-fluid-v1.5.1"
# 启动命令
#start_cmd="python -u launch.py train.py \
#           --src_vocab_fpath ./wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#           --trg_vocab_fpath ./wmt16_ende_data_bpe/vocab_all.bpe.32000 \
#           --special_token '<s>' '<e>' '<unk>' \
#           --train_file_pattern ./wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
#           --token_delimiter ' ' \
#           --use_token_batch True \
#           --batch_size 2048 \
#           --nccl_comm_num 3 \
#           --use_hierarchical_allreduce True \
#           --fuse True \
#           --hierarchical_allreduce_inter_nranks 8 \
#           --use_experimental_executor True \
#           --enable_backward_op_deps True \
#           --batch_size 2048 \
#           --sort_type pool \
#           --update_method nccl2 \
#           --pool_size 200000 \
#           n_head 8 \
#           d_model 512 \
#           d_inner_hid 2048 \
#           prepostprocess_dropout 0.1"
#start_cmd="python launch.py train.py --fp16=False --data_dir=./fast_resnet_data --nccl_comm_num=2 \
#    --use_hierarchical_allreduce=False --fuse=True  --enable_backward_op_deps=True \
#    --hierarchical_allreduce_inter_nranks=8 --log_period=5 --profile=True \
#    --start_profile_batch=20 --stop_profile_batch=26 --num_threads=4"

start_cmd="python run.py"

paddlecloud job --server ${server} \
        --port ${port} \
        --user-ak ${user_ak} \
        --user-sk ${user_sk} \
        train --job-name ${job_name} \
        --job-conf config.ini \
        --start-cmd "${start_cmd}" \
        --files end_hook.sh before_hook.sh ../*.py ../utils/*.py \
        --cluster-name ${cluster_name} \
        --job-version ${job_version}  \
        --k8s-gpu-type ${k8s_gpu_type} \
        --k8s-gpu-cards $gpus_per_node \
        --k8s-wall-time ${k8s_wall_time} \
        --k8s-memory ${k8s_memory} \
        --k8s-cpu-cores 35 \
        --k8s-priority=${k8s_priority} \
        --k8s-trainers ${k8s_trainers} \
        --is-standalone 1
#        --distribute-job-type NCCL2
