unset http_proxy
unset https_proxy

PADDLE_PSERVERS_IP_PORT_LIST=127.0.0.1:6017,127.0.0.1:6018 \
PADDLE_TRAINERS_NUM=2 \
TRAINING_ROLE=PSERVER \
POD_IP=127.0.0.1 \
PADDLE_PORT=6017 \
python distributed_train.py 2> server0.elog > server0.stdlog &

sleep 3

PADDLE_PSERVERS_IP_PORT_LIST=127.0.0.1:6017,127.0.0.1:6018 \
PADDLE_TRAINERS_NUM=2 \
TRAINING_ROLE=PSERVER \
POD_IP=127.0.0.1 \
PADDLE_PORT=6018 \
python distributed_train.py 2> server1.elog > server1.stdlog &

sleep 3

PADDLE_PSERVERS_IP_PORT_LIST=127.0.0.1:6017,127.0.0.1:6018 \
PADDLE_TRAINERS_NUM=2 \
TRAINING_ROLE=TRAINER \
PADDLE_TRAINER_ID=0 \
python distributed_train.py 2> worker0.elog > worker0.stdlog &

sleep 3

PADDLE_PSERVERS_IP_PORT_LIST=127.0.0.1:6017,127.0.0.1:6018 \
PADDLE_TRAINERS_NUM=2 \
TRAINING_ROLE=TRAINER \
PADDLE_TRAINER_ID=1 \
python distributed_train.py 2> worker1.elog > worker1.stdlog &
