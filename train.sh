#!/bin/sh

export GOOGLE_APPLICATION_CREDENTIALS="key.json"
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
LOGDIR="assets/out/models/exp11"
POLICY_VERSION=3
ENV_TYPE="maze"
TIMESTEPS=1000000
MAX_EPISODE_SIZE=2000
LEARNING_TYPE="explore"
N_STEPS=4

python3 train.py \
    --logdir $LOGDIR \
    --timesteps $TIMESTEPS \
    --max_episode_size $MAX_EPISODE_SIZE \
    --learning_type $LEARNING_TYPE \
    --policy_version $POLICY_VERSION \
    --env_type $ENV_TYPE \
    --n_steps $N_STEPS
