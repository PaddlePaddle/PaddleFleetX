#!/bin/bash
set -ux

if [[ $# == 1 ]]; then
    job_conf=$1
    source ${job_conf}
elif [[ $# > 1 ]]; then
    echo "usage: sh $0 [job_conf]"
    exit -1
fi

export FLAGS_sync_nccl_allreduce=1
export FLAGS_fuse_parameter_memory_size=64

mkdir -p ${save_path}

if [[ ${log_dir:-""} != "" ]]; then
    mkdir -p ${log_dir}
    distributed_args="${distributed_args:-} --log_dir ${log_dir}"
fi

# ulimit -c unlimited

fleetrun \
    ${distributed_args:-} \
    ./knover/scripts/train.py \
    --use_k8s ${use_k8s:-"false"} \
    --is_distributed true \
    --model ${model:-"Plato"} \
    --task ${task:-"DialogGeneration"} \
    --vocab_path ${vocab_path} \
    --specials_path ${specials_path:-""} \
    --do_lower_case ${do_lower_case:-"false"} \
    --spm_model_file ${spm_model_file} \
    --init_pretraining_params ${init_params:-""} \
    --init_checkpoint ${init_checkpoint:-""} \
    --train_file ${train_file} \
    --valid_file ${valid_file} \
    --data_format ${data_format:-"raw"} \
    --file_format ${file_format:-"file"} \
    --config_path ${config_path} \
    ${train_args:-} \
    --in_tokens ${in_tokens:-"false"} \
    --batch_size ${batch_size:-8192} \
    --learning_rate ${lr} \
    --warmup_steps ${warmup_steps:-0} \
    --weight_decay ${weight_decay:-0.0} \
    --use_amp ${use_amp:-"true"} \
    --use_recompute ${use_recompute:-"false"} \
    --num_epochs ${num_epochs} \
    --log_steps ${log_steps} \
    --validation_steps ${validation_steps} \
    --save_steps ${save_steps} \
    --save_path ${save_path} \
    --random_seed ${random_seed:-11}
exit_code=$?

exit $exit_code
