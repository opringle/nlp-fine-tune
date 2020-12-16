# Fine tune BERT on GCP

Leverage GCP Ai Platform to speed up your ML workflow.

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
    pip install torch && pip install -r requirements.txt
```
- Ensure you have the following GCP roles:
  - `cloudbuild.builds.editor` - build and push container images using Cloud Build
  - `ml.developer` - submit training/inference jobs to Ai Platform
- Create a storage bucket for the training data and model artifacts, logs etc.
```bash
  gsutil mb gs://nlp-fine-tuner
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

## Results training Roberta large (334M parameters)

| batch size |        gpu        | number gpus |  worker type  | CPU utilization | Memory utilization | GPU memory utizilization | GPU utilization | examples per second |
|------------|-------------------|-------------|---------------|-----------------|--------------------|--------------------------|-----------------|---------------------|
|     1      | NVIDIA_TESLA_K80  |      1      | n1-standard-4 |        25%      |          36%       |            72%           |      100%       |           1         | 
|     2      | NVIDIA_TESLA_K80  |      1      | n1-standard-4 |        25%      |          36%       |            88%           |      100%       |          1.5        | 
|     4      | NVIDIA_TESLA_K80  |      1      | n1-standard-4 |         -       |           -        |             -            |        -        |          OOM        | 
|     4      | NVIDIA_TESLA_V100 |      1      | n1-standard-4 |        26%      |          46%       |            94%           |      92%        |          9          | 
|     6      | NVIDIA_TESLA_V100 |      1      | n1-standard-4 |        26%      |          46%       |            94%           |      92%        |          OOM        |
|     8      | CLOUD_TPU_V3      |      1      | n1-standard-4 |        23%      |          10%       |             -            |        -        |          ---        | 

## ToDo

- Single V3 TPU training


- Multi GPU cloud training with DistributedDataParallel
- Multi node, multi GPU training
- Add model parameters to docker image so they don't need to be downloaded
- Train on data uploaded to GCS
- Multi node multi GPU training with DistributedDataParallel
- Visualize training with tensorboard
- Use half precision training (FP16) to increase throughput
- View GPU utliziation during training

- Multi-node multi GPU cloud training?
- Single TPU cloud training
- Multi TPU cloud training
- Multi-node multi TPU cloud training
- Deploy the model for scalable prediction
- Plot training time vs cost for different runs
- Automate deployment

## Notes

- With 8 V100 GPUS I can train on 72 examples per second (30 minutes per epoch on AG News 120k training set)
- With multi node multi GPU training, I can train a little faster but I'm still severly limited by GPU memory
- Training on V3 TPUs will allow for much larger batch sizes and throughput

The application downloads the model before running. This makes it very slow.

You can trigger a cloud build from a push to a specific branch of your repo. This means when you update master your infrastructure automatically updates without you doing anything.

We could have a machine learning application running in a container on cloud run. Then have the container built and pushed whenever master changes. - https://cloud.google.com/cloud-build/docs/deploying-builds/deploy-cloud-run

Fine tuning language models can be slow because the models have so many parameters that only small batches can be used during fine-tuning.

Hugging face have written examples for all this here - https://github.com/huggingface/transformers/tree/master/examples/text-classification

