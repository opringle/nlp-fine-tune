#!/bin/bash
# This scripts performs cloud fine tuning of a language model

echo "Training Cloud ML model"

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=nlp-fine-tuner
CONFIG_FILE=training_configs/gpu_config.yaml

DATE=$(date '+%Y%m%d_%H%M%S')
JOB_NAME=text_classifier_${DATE}

JOB_DIR=gs://${BUCKET_NAME}/trainer
REGION=us-central1
TRAIN_DATA_FILE=gs://${BUCKET_NAME}/data/df.pickle

set -v

gcloud ai-platform jobs submit training "${JOB_NAME}" \
  --region $REGION \
  --config $CONFIG_FILE \
  -- \
  --job-dir $JOB_DIR \
  --train-data-file $TRAIN_DATA_FILE \
  --pretrained_model_path /root/roberta/ \
  --batch-size 1 \
  --max_epochs 1 \
  --gpus 2 \
  --accelerator ddp \
  --precision 16 \
