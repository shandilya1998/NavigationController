#!/bin/sh

export GOOGLE_APPLICATION_CREDENTIALS="key.json"
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
LOGDIR="assets/out/models/exp3"
TIMESTEPS=1000000
BATCH_SIZE=128
MAX_EPISODE_SIZE=2500
LEARNING_TYPE="imitate"

python3 train.py \
    --logdir $LOGDIR \
    --timesteps $TIMESTEPS \
    --batch_size $BATCH_SIZE \
    --max_episode_size $MAX_EPISODE_SIZE \
    --learning_type $LEARNING_TYPE
