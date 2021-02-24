export PADDLE_TRAINER_ENDPOINTS="127.0.0.1:8200"
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:8500"
export PADDLE_TRAINERS_NUM=1
export POD_IP=127.0.0.1
export PADDLE_PORT=$2
export PADDLE_TRAINER_ID=$3

if [ "$1" = "TRAINER" ];then
    echo "TRAINER !"
    export TRAINING_ROLE=TRAINER
    python -u train.py

else
    echo "PSERVER !"
    export TRAINING_ROLE=PSERVER
    python -u train.py
fi

exit 0
