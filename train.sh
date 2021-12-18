#!/bin/sh

export GOOGLE_APPLICATION_CREDENTIALS="key.json"
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
LOGDIR="assets/out/models/exp18"
POLICY_VERSION=6
ENV_TYPE="maze"
TIMESTEPS=1000000
MAX_EPISODE_SIZE=250
LEARNING_TYPE="explore"
HISTORY_STEPS=5
TASK_VERSION=3
N_STEPS=0
LAMBDA=1

python3 train.py \
    --logdir $LOGDIR \
    --timesteps $TIMESTEPS \
    --max_episode_size $MAX_EPISODE_SIZE \
    --learning_type $LEARNING_TYPE \
    --policy_version $POLICY_VERSION \
    --env_type $ENV_TYPE \
    --history_steps $HISTORY_STEPS \
    --task_version $TASK_VERSION \
    --n_steps $N_STEPS \
    --lmbda $LAMBDA
