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
