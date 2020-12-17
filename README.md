# Fine tune BERT on GCP

Leverage GCP as a training job management platform to speed up your ML workflow.

## Prerequisites

- [install & configure Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [install docker](https://docs.docker.com/get-docker/)
- Set docker system memory >= 4Gb 
- [install pyenv](https://realpython.com/intro-to-pyenv/)
- Create & activate python virtual environment
```bash
    pyenv virtualenv 3.6.9 nlp && pyenv local nlp
```
- Install tensorflow and other required python packages
```bash
    pip install tensorflow==2.3.0 && pip install -r requirements.txt
```
- Ensure you have the following GCP roles:
  - `cloudbuild.builds.editor` - build and push container images using Cloud Build
  - `ml.developer` - submit training/inference jobs to Ai Platform
- Create a storage bucket for the training data and model artifacts, logs etc.
```bash
  gsutil mb gs://nlp-fine-tuner
```
- Create a tpu service account

```bash
gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID
```
The command returns a Cloud TPU Service Account with following format:
```bash
  service-PROJECT_NUMBER@cloud-tpu.iam.gserviceaccount.com
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
    --train-data-file data/df.pickle \
    --job-dir . \
    --epochs 1 \
    --batch-size 1
```

- Upload your image to Google Container Registry (GCR), using Cloud Build to build and push the image faster.

```bash
  gcloud builds submit --config cloudbuild.yaml .
```

- Train the model on a cloud GPU

```bash
  bash scripts/train-cloud-gpu.sh
```

## Why use TPUs?

Results training Roberta large (334M parameters)

| batch size |        gpu        | number devices |  worker type  | CPU utilization | Memory utilization | GPU memory utizilization | GPU utilization | examples per second |
|------------|-------------------|-------------|---------------|-----------------|--------------------|--------------------------|-----------------|---------------------|
|1|NVIDIA_TESLA_K80|1|n1-standard-4|25%|36%|72%|100%|0.25| 
|2|NVIDIA_TESLA_K80|2|n1-standard-4|25%|36%|72%|100%|0.44| 
|8|NVIDIA_TESLA_V100|2|n1-standard-4|-|-|-|-|OOM|
|16|CLOUD_TPU_V2|8|n1-highcpu-16||||||
|16|CLOUD_TPU_V3|8|n1-highcpu-16||||||

## Learnings

- It is possible to use pytorch with TPUs, however, it's a pain in the arse. You need to install the xla library and make significant modifications to the code. Using Keras and the strategy scope you can run in any configuration.
- Tensorflow is slower than Pytorch by a factor of 4 on GPUs for some jobs!
- Pytorch lightning allows you to minimize the reconfiguration work and still use pytorch :)

## ToDo

- Single V2/3 TPU training on 8 cores
- Support training on X% of the data
- Preprocess AG News to GCS
- Download data to docker image so it doesn't need to be downloaded
- Pytorch lightning refactor
  - Upgrade to CUDA 11
  - Upgrade to latest pytorch
  
- Add model parameters to docker image so they don't need to be downloaded
- Package training application with local testing
- Visualize training with tensorboard
- Use half precision training (FP16) to increase throughput
