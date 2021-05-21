#!/bin/bash
################################################################################
# Expand embedding: `embedding_name` from [old_size, hidden_size] -> [new_size, hidden_size]
# Retain the first `old_size` params.
################################################################################

PYTHONPATH=.

PARAM_PATH="/path/to/model"
python -u \
    ./tools/expand_embedding.py \
    --param_path ${PARAM_PATH} \
    --save_path ${PARAM_PATH} \
    --embedding_name pos_embedding \
    --embedding_new_size 512
