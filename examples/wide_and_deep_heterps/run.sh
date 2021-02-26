#!/bin/bash
# Usage: sh run.sh

mkdir -p ./log && hostname

# 1. export environment config
export GLOG_v=1
source ./heterps.bashrc

# 2. run server 
# port must be the same in run_psgpu.sh
sh run_psgpu.sh PSERVER 8500 0 &

# 3. run worker
sh run_psgpu.sh TRAINER 8200 0 &
