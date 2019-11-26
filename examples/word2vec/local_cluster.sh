#!/bin/bash
echo "WARNING: This script only for run PaddlePaddle Fluid on one node"
echo "Running 2X2 Parameter Server model"

if [ ! -d "./log" ]; then
  mkdir ./log
  echo "Create log floder for store running log"
fi

# environment variables for fleet distribute training
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:36011,127.0.0.1:36012"
export PADDLE_TRAINERS_NUM=2
export CPU_NUM=2

## environment variables for pserver
export TRAINING_ROLE=PSERVER
export POD_IP=127.0.0.1

paddle_pserver_nums=2
paddle_pserver_ports=(36011 36012) 
for((i=0;i<${paddle_pserver_nums};i++))
do
    export PADDLE_PORT=${paddle_pserver_ports[$i]}
    echo "PADDLE WILL START PSERVER "${PADDLE_PORT}
    python -u dist_train_example.py &> ./log/pserver.$i.log &
done

## environment variables for trainer
export TRAINING_ROLE=TRAINER
for((i=0;i<${PADDLE_TRAINERS_NUM};i++))
do
    export PADDLE_TRAINER_ID=${i}
    echo "PADDLE WILL START TRAINER "${i}
    python -u dist_train_example.py &> ./log/trainer.$i.log &
done
