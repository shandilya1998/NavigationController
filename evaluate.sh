#!/bin/sh

LOGDIR="assets/out/models/exp7"
MODEL_FILE="best_model"
MAX_EPISODE_SIZE=1500

python3 evaluate.py \
    --logdir $LOGDIR \
    --model_file $MODEL_FILE \
    --max_episode_size $MAX_EPISODE_SIZE
