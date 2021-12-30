#!/bin/sh

export GOOGLE_APPLICATION_CREDENTIALS="key.json"
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
#LOGDIR="/content/drive/MyDrive/CNS/exp22"
LOGDIR="assets/out/models/exp22"
ENV_TYPE="maze"
TIMESTEPS=1000000
MAX_EPISODE_SIZE=1000
LEARNING_TYPE="explore"
HISTORY_STEPS=15
TASK_VERSION=1
N_STEPS=0
LAMBDA=1
MODEL_TYPE='recurrent'

python3 train.py \
    --logdir $LOGDIR \
    --timesteps $TIMESTEPS \
    --max_episode_size $MAX_EPISODE_SIZE \
    --learning_type $LEARNING_TYPE \
    --env_type $ENV_TYPE \
    --history_steps $HISTORY_STEPS \
    --task_version $TASK_VERSION \
    --n_steps $N_STEPS \
    --lmbda $LAMBDA \
    --model_type $MODEL_TYPE
