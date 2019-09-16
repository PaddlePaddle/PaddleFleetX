#!/bin/bash

if [ ! -d ./evals ]; then
    mkdir ./evals
    echo "mkdir ./evals to save evaluate results and logs"
fi

if [ ! -d ./evals/logs ]; then
    mkdir ./evals/logs
    echo "mkdir ./evals/logs to save evaluate logs"
fi

if [ ! -d ./evals/results ]; then
    mkdir ./evals/results
    echo "mkdir ./evals/results to save evaluate results"
fi

modes_need_to_evaluate=(sync_dataset)
checkpoints_path=(/work/tensorflow/ctr/dataset/output/checkpoint/sync) # /work/tensorflow/ctr/reader/output/checkpoint/local)
nums=1
result_path=./evals/results

for((i=0;i<${nums};i++));
do
    mode=${modes_need_to_evaluate[$i]}
    path=${checkpoints_path[$i]}
    python -u eval.py --task_mode=${mode} --checkpoint_path=${path} --result_path=${result_path} &> ./evals/logs/${mode}.log &
done
