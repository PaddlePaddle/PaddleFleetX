#!/bin/bash

#export FLAGS_communicator_send_queue_size=4
#export FLAGS_communicator_max_merge_var_num=4
#export FLAGS_communicator_merge_sparse_grad=0
# start pserver0
python dist_ctr.py \
    --is_local 0 \
    --role pserver \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --current_endpoint 127.0.0.1:6000 \
    --trainers 2 \
    > pserver0.log 2>&1 &

# start pserver1
python dist_ctr.py \
    --is_local 0 \
    --role pserver \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --current_endpoint 127.0.0.1:6001 \
    --trainers 2 \
    > pserver1.log 2>&1 &

# start trainer0
python dist_ctr.py \
    --is_local 0 \
    --role trainer \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --trainers 2 \
    --trainer_id 0 \
    > trainer0.log 2>&1 &

# start trainer1
python dist_ctr.py \
    --is_local 0 \
    --role trainer \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --trainers 2 \
    --trainer_id 1 \
    > trainer1.log 2>&1 &
