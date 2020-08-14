#!/bin/bash
echo "WARNING: This script only for run PaddlePaddle Fluid on one node"
echo "Running 5X5 Parameter Server model"

if [ ! -d "./model" ]; then
  mkdir ./model
  echo "Create model folder for store infer model"
fi

if [ ! -d "./result" ]; then
  mkdir ./result
  echo "Create result folder for store training result"
fi

if [ ! -d "./log" ]; then
  mkdir ./log
  echo "Create log floder for store running log"
fi

# environment variables for fleet distribute training
export PADDLE_TRAINER_ID=0
export PADDLE_TRAINERS_NUM=4
export PADDLE_PORT=36011,36012,36013,36014
export PADDLE_PSERVERS=127.0.0.1
export POD_IP=127.0.0.1
export CPU_NUM=16
export OUTPUT_PATH="output"
export SYS_JOB_ID="local_cluster"
#export FLAGS_communicator_send_queue_size=1
#export FLAGS_communicator_thread_pool_size=5
export FLAGS_rpc_retry_times=3
export FLAGS_communicator_max_merge_var_num=16
export FLAGS_communicator_fake_rpc=0

export PADDLE_PSERVER_PORTS=36011,36012,36013,36014
export PADDLE_PSERVER_PORT_ARRAY=(36011 36012 36013 36014)

export PADDLE_PSERVER_NUMS=4
export PADDLE_TRAINERS=4

train_method=$1
sync_mode=$2

export GLOG_v=0
export GLOG_logtostderr=1

export TRAINING_ROLE=PSERVER
for((i=0;i<$PADDLE_PSERVER_NUMS;i++))
do
    cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
    echo "PADDLE WILL START PSERVER "$cur_port
    PADDLE_TRAINER_ID=$i
    python -u model.py --is_local=0 --is_${train_method}_train=True --is_local_cluster=True --sync_mode=${sync_mode} &> ./log/pserver.$i.log &
done

export TRAINING_ROLE=TRAINER
for((i=0;i<$PADDLE_TRAINERS;i++))
do
    echo "PADDLE WILL START Trainer "$i
    PADDLE_TRAINER_ID=$i
    python -u model.py --is_local=0 --is_${train_method}_train=True --is_local_cluster=True --sync_mode=${sync_mode} &> ./log/trainer.$i.log &
done

