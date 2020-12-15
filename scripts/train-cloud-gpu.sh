#!/bin/bash
# This scripts performs cloud fine tuning of a language model

echo "Training Cloud ML model"

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=nlp-fine-tuner
CONFIG_FILE=training_configs/gpu_config.yaml

IMAGE_REPO_NAME=nlp
IMAGE_TAG=latest
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

DATE=$(date '+%Y%m%d_%H%M%S')
JOB_NAME=bp_$(date +%Y%m%d_%H%M%S)

JOB_DIR=gs://${BUCKET_NAME}/trainer
REGION=us-central1
TRAIN_DATA_FILE=gs://${BUCKET_NAME}/data/df.pickle

set -v

gcloud ai-platform jobs submit training "${JOB_NAME}" \
  --master-image-uri $IMAGE_URI \
  --region $REGION \
  --config $CONFIG_FILE \
  -- \
  --job-dir $JOB_DIR \
  --train-data-file $TRAIN_DATA_FILE \
  --epochs 5 \
  --batch-size 1 \