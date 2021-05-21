#!/bin/bash
set -eux

if [[ $# != 1 ]]; then
    echo "input: job_conf"
    exit -1
fi

job_conf=$1
source ${job_conf}

if [[ ${nodes:-1} -gt 1 ]]; then
    is_standalone=0
else
    is_standalone=1
fi

paddlecloud job \
    --ak ${user_ak} \
    --sk ${user_sk} \
    train \
    --job-name $task_name \
    --group-name $queue \
    --job-conf ${job_conf} \
    --job-version ${job_version:-"paddle-fluid-v1.7.1"} \
    --job-tags ${job_tags:-""} \
    --job-remark ${job_remark:-""} \
    --image-addr ${image_addr:-""} \
    --file-dir . \
    --start-cmd "./scripts/paddlecloud/k8s_job.sh ${job_conf}" \
    --k8s-priority ${priority:-"high"} \
    --k8s-gpu-cards ${cards:-8} \
    --k8s-trainers ${nodes:-1} \
    --is-standalone ${is_standalone} \
    --is-auto-over-sell ${oversell:-0} \
    --permission ${permission:-"private"} \
    --k8s-wall-time 00:00:00
