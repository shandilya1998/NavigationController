#!/bin/sh

export GOOGLE_APPLICATION_CREDENTIALS="key.json"
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
LOGDIR="assets/out/models"
TIMESTEPS=1000
BATCH_SIZE=3
MAX_EPISODE_SIZE=10
LEARNING_TYPE="explore"

python3 train.py \
    --logdir gs://$BUCKET_NAME \
    --timesteps $TIMESTEPS \
    --batch_size $BATCH_SIZE \
    --max_episode_size $MAX_EPISODE_SIZE \
    --learning_type $LEARNING_TYPE
