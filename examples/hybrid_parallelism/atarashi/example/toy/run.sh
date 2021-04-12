#!/bin/sh
#export PYTHONPATH=./atarashi-paddle/:./bin/atarashi/protos/:$PYTHONPATH

python=./app/bin/python3

#python test.py

echo "Using gpu env ${CUDA_VISIBLE_DEVICES}"
#${python} ./dataset_test.py \

${python} ./toy/main.py \
    --train_data_dir ./data_ernie/train/ \
    --eval_data_dir  ./data_ernie/dev/ \
    --max_seqlen 128 \
    --run_config '{
        "batch_size": 32,
        "model_dir": "./models/toy_10",
        "max_steps": 60000,
        "save_steps": 1000,
        "log_steps": 10,
        "skip_steps": 10, # comment
        "eval_steps": 100,
        "shit": 0
    }' \
    --hparam '{
        "hidden_size": 256,
        "vocab_size": 50000,
        "embedding_size": 256,
        "num_layers": 3,
        "learning_rate": 0.0001
    }' \
    --vocab_size 50000  \
    --vocab_file ernie_model/vocab.txt 


