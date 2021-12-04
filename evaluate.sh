#!/bin/sh

LOGDIR="assets/out/models/exp7"
MODEL_FILE="rl_model_27000_steps"
MAX_EPISODE_SIZE=1500

python3 evaluate.py \
    --logdir $LOGDIR \
    --model_file $MODEL_FILE \
    --max_episode_size $MAX_EPISODE_SIZE
