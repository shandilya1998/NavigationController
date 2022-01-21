#!/bin/sh

LOGDIR="assets/out/models/exp22"
MODEL_FILE="rl_model_96048_steps"
MAX_EPISODE_SIZE=750
HISTORY_STEPS=15

python3 evaluate.py \
    --logdir $LOGDIR \
    --model_file $MODEL_FILE \
    --max_episode_size $MAX_EPISODE_SIZE \
    --history_steps $HISTORY_STEPS
