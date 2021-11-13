#!/bin/sh

export GOOGLE_APPLICATION_CREDENTIALS="key.json"
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
LOGDIR="assets/out/models"
TIMESTEPS=1000000
<<<<<<< HEAD
BATCH_SIZE=2
=======
BATCH_SIZE=128
>>>>>>> 34b70fa9c3b71ad66875ea41dcdde76265a94ae9
MAX_EPISODE_SIZE=1500
LEARNING_TYPE="explore"

python3 train.py \
    --logdir $LOGDIR \
    --timesteps $TIMESTEPS \
    --batch_size $BATCH_SIZE \
    --max_episode_size $MAX_EPISODE_SIZE \
    --learning_type $LEARNING_TYPE
