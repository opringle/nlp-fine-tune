# Fine tune large models on GCP

This repo shows how to leverage GCP Ai Platform Training, pytorch and pytorch-lightning to train models on CPU, GPUs or TPUs with minimal code changes. It is intended to provide a template for deep learning projects.

## Prerequisites

- [install & configure Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [install docker](https://docs.docker.com/get-docker/)
- Set docker system memory >= 4Gb 
- [install pyenv](https://realpython.com/intro-to-pyenv/)
- Create & activate python virtual environment
```bash
    pyenv virtualenv 3.8.6 nlp && pyenv local nlp
```
- Install pytorch and other required python packages
```bash
    pip install torch==1.7.1 && pip install -r requirements.txt
```
- Ensure you have the following GCP roles:
  - `cloudbuild.builds.editor` - build and push container images using Cloud Build
  - `ml.developer` - submit training/inference jobs to Ai Platform
- Create a storage bucket for the training data and model artifacts, logs etc.
```bash
  gsutil mb gs://nlp-fine-tuner
```
- Get the tpu service account associated with your project
```bash
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" https://ml.googleapis.com/v1/projects/${PROJECT_ID}:getConfig
```
- Assign roles so the service account can read/write to GCS etc (editor is lazy way to do it)
```bash
  gcloud projects add-iam-policy-binding $PROJECT_ID --member serviceAccount:$TPU_SERVICE_ACCOUNT --role roles/editor
```

## Fine-tune a language model

### Test code locally on CPU

```bash
  bash scripts/train-local.sh
```

### Cloud

- The training application uses a docker container to run on GCP. Test your application locally before submitting the job.

```bash
  docker build -t nlp .
```

```bash
  docker run nlp \
    --job-dir . \
    --train-data-file data/df.pickle \
    --batch-size 1 \
    --max_epochs 10 \
    --pretrained_model_path /root/roberta/ \
```

- Upload your image to Google Container Registry (GCR), using Cloud Build to build and push the image faster.

```bash
  gcloud builds submit --config cloudbuild.yaml .
```

- Train the model on GPUs in the cloud

```bash
  bash scripts/train-cloud-gpu.sh
```

- Train the model on a cloud TPU

```bash
  bash scripts/train-cloud-tpu.sh
```

## Why use TPUs?

Results training Roberta large (334M parameters)

| batch size |        gpu        | number devices |  worker type  | CPU utilization | Memory utilization | GPU memory utizilization | GPU utilization | examples per second |
|------------|-------------------|-------------|---------------|-----------------|--------------------|--------------------------|-----------------|---------------------|
|1|NVIDIA_TESLA_K80|1|n1-standard-4|25%|36%|72%|100%|0.25| 
|2|NVIDIA_TESLA_K80|2|n1-standard-4|25%|36%|72%|100%|0.44| 
|8|NVIDIA_TESLA_V100|2|n1-standard-4|-|-|-|-|OOM|
|16|CLOUD_TPU_V2|8|n1-highcpu-16|||||2400|
|128|CLOUD_TPU_V3|8|n1-highcpu-16||||||

## Learnings

- It is possible to use pytorch with TPUs, however, it's a pain in the arse. You need to install the xla library and make significant modifications to the code. 
- Using Keras and the strategy scope you can run in any configuration.
- Tensorflow is slower than Pytorch by a factor of 4 on GPUs for some jobs!
- Pytorch lightning allows you to minimize the reconfiguration work required to train on CPUs, GPUs or TPUs. Allowing the researcher to focus on code not infrastructure.

## ToDo

- Download data to docker image so it doesn't need to be downloaded when application starts
  - Should I preprocess then upload tensors to GCS? Takes a while to process the dataset...
- Refactor for pytorch
- Train pytorch lightning model on single GPU
- Train pytorch lightning model on multiple GPUs
- Single V2/3 TPU training on 8 cores
- Support training on X% of the data
- Upgrade to CUDA 11
- Package training application with local testing as per google groups
- Visualize training with tensorboard
