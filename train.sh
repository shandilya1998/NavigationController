#!/bin/sh

export GOOGLE_APPLICATION_CREDENTIALS="key.json"
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
LOGDIR="assets/out/models/exp6"
TIMESTEPS=1000000
MAX_EPISODE_SIZE=1500
LEARNING_TYPE="explore"

python3 train.py \
    --logdir $LOGDIR \
    --timesteps $TIMESTEPS \
    --max_episode_size $MAX_EPISODE_SIZE \
    --learning_type $LEARNING_TYPE
