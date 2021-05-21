#!/bin/bash
set -eux

if [[ $# == 1 ]]; then
    job_conf=$1
    source ${job_conf}
elif [[ $# > 1 ]]; then
    echo "usage: sh $0 [job_conf]"
    exit -1
fi

export FLAGS_sync_nccl_allreduce=1
export FLAGS_fuse_parameter_memory_size=64


# Process NSP model(for reranking in dialogue generation task).
if [[ ${nsp_init_params:-} != "" ]]; then
    if [[ ! -e "${nsp_init_params}/__model__" ]]; then
        python -m \
            knover.scripts.save_inference_model \
            --model NSPModel \
            --task NextSentencePrediction \
            --vocab_path ${vocab_path} \
            --init_pretraining_params ${nsp_init_params} \
            --spm_model_file ${spm_model_file} \
            --inference_model_path ${nsp_init_params} \
            ${save_args:-} \
            --config_path ${config_path}
    fi
    infer_args="--nsp_inference_model_path ${nsp_init_params} ${infer_args:-}"
fi

python -m \
    knover.scripts.self_chat \
    --model ${model} \
    --vocab_path ${vocab_path} \
    --spm_model_file ${spm_model_file} \
    --config_path ${config_path} \
    --init_pretraining_params ${init_params} \
    --batch_size ${batch_size:-2} \
    --num_episode ${num_episode:-10} \
    --num_turn ${num_turn:-10} \
    ${infer_args:-}
