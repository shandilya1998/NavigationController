#!/bin/sh

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform

echo "Project ID: $PROJECT_ID"
echo "Project Bucket: $BUCKET_NAME"

REGION="$(gcloud config get-value compute/region)"
echo "Project Region: $REGION"

# Commands to create a bucket and store error in a variable
# { var="$( { gsutil mb -l $REGION -p $PROJECT_ID gs://$BUCKET_NAME; } 2>&1 1>&3 3>&- )"; } 3>&1;
# echo "Test"
# echo $var

IMAGE_REPO_NAME=navigation_controller_container
IMAGE_TAG=navigation_controller_pytorch
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
JOB_NAME=experiment_$(date +%Y%m%d_%H%M%S)

docker build -f docker/Dockerfile -t $IMAGE_URI ./
docker push $IMAGE_URI

# docker run $IMAGE_URI --logdir assets/logs --timesteps 1000 \
# --batch_size 1 --max_episode_size 100

gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  --config docker/config.yaml \
  -- \
  --logdir=gs://$BUCKET_NAME \
  --timesteps=1000000 \
  --batch_size=128 \
  --max_episode_size=5000

gcloud ai-platform jobs stream-logs $JOB_NAME
