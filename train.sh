#!/bin/sh

LOGDIR="assets/out/models"
TIMESTEPS=1000
BATCH_SIZE=1
MAX_EPISODE_SIZE=10
LEARNING_TYPE="explore"

python3 train.py \
    --logdir $LOGDIR \
    --timesteps $TIMESTEPS \
    --batch_size $BATCH_SIZE \
    --max_episode_size $MAX_EPISODE_SIZE \
    --learning_type $LEARNING_TYPE
