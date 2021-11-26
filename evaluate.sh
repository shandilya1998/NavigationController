#!/bin/sh

LOGDIR="assets/out/models/exp1"
MODEL_FILE="rl_model_900000_steps"

python3 evaluate.py \
    --logdir $LOGDIR \
    --model_file $MODEL_FILE
