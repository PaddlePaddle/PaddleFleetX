#!/bin/bash
for step in {0..10} ; do
    echo $step
    pkill -f python 
    sleep 10s

    ./scripts/train_gpu.sh 
    sleep 20m
done
