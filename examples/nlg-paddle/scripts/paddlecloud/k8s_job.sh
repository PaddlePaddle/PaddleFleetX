#!/bin/bash
################################################################################
# Run job on k8s cluster.
################################################################################
set -eux

if [[ $# != 1 ]]; then
    echo "input: job_conf"
    exit -1
fi

job_conf=$1
source ${job_conf}

mkdir -p log
mkdir -p output

source ./scripts/setup.sh &> log/setup.log

# Prepare running envrionment.
if [[ ${process_env_script:-""} != "" ]]; then
    source ${process_env_script}
fi

export use_k8s="true"
export log_dir="./log"
export save_path="./output"
export random_seed=$RANDOM
export PATH=$PWD/python/bin:$PATH
sh ${job_script:-"./scripts/distributed/train.sh"} ${job_conf}
