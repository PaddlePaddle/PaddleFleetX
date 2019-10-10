#!/bin/bash
export PADDLE_PSERVERS_IP_PORT_LIST=127.0.0.1:36001,127.0.0.1:36002,127.0.0.1:36003,127.0.0.1:36004,127.0.0.1:36005
export PADDLE_WORKERS_IP_PORT_LIST=127.0.0.1:36006,127.0.0.1:36007,127.0.0.1:36008,127.0.0.1:36009,127.0.0.1:36010
trainer_nums=5
pserver_nums=5

mode=$1
for((i=0;i<${pserver_nums};i++)) 
do
    export TRAINING_ROLE=PSERVER
    export PADDLE_TRAINER_ID=$i
    python -u word2vec_distribute.py --dist_mode=${mode} &> ./log/${mode}/pserver.$i.log &
done

for((i=0;i<${trainer_nums};i++))
do
    export TRAINING_ROLE=TRAINER
    export PADDLE_TRAINER_ID=$i
    python -u word2vec_distribute.py --dist_mode=${mode} &> ./log/${mode}/worker.$i.log &
done
