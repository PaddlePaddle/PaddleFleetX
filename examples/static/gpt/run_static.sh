#!/bin/bash
set -eux

if [[ $# < 2 ]]; then
    echo "input: job_type (mp|dp|dpmp|auto_mp|auto_dp|auto_dpmp) cards"
    exit -1
fi

job_type=$1
gpu_card=$2
debug=false

if [[ $# == 3 ]]; then
    debug=true
fi

echo ${debug}

echo ${gpu_card}
# echo ${job_type}

if [[ ${job_type} == auto* ]];then
    job_conf="run_auto_parallel_"${job_type:5}".sh"
else
    job_conf="run_hybrid_parallel_"${job_type}".sh"
fi

# echo ${job_conf}
sh ${job_conf} ${gpu_card} ${debug}
