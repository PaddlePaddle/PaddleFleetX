#!/bin/bash
for step in {0..10} ; do
    echo $step
    pkill -f python 
    pkill -f train_gpu.sh
    sleep 10s

    m=$(( step % 2))
    num_cards=8
    if (( m == 0 )) ; then
        num_cards=6
    fi
    echo "num_card:" $num_cards

    nohup ./scripts/train_gpu.sh -num_cards $num_cards > train.log 2>&1  &

    sleep 20m
done
