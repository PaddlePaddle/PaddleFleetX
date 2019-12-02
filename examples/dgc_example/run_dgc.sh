#!/bin/bash

set -eu

# print debug info
export GLOG_v=1

python -m paddle.distributed.launch --selected_gpus="0,1" --log_dir=log \
    train_with_fleet.py --use_dgc True
