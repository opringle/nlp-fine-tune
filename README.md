# Fine tune BERT on GCP

Multi-node-multi-TPU fine-tuning & deployment of BERT on GCP.

## Prerequisites

- [install & configure Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [install docker](https://docs.docker.com/get-docker/)
  - configure docker to use Container Registry - `gcloud auth configure-docker`
- [install pyenv](https://realpython.com/intro-to-pyenv/)
- create & activate python virtual environment
```bash
    pyenv virtualenv 3.8.6 bert && pyenv local bert
```
- install required python packages - `pip install -r requirements.txt`
- ensure you have the following GCP roles:
  - `cloudbuild.builds.editor` - build and push container images using Cloud Build
  - `ml.developer` - submit training/inference jobs to Ai Platform

## Training the model

### Locally

```bash
  bash scripts/train-local.sh
```



## ToDo

- Single GPU cloud training
- Multi GPU cloud training
- Multi-node multi GPU cloud training
- Multi-node multi TPU cloud training
- Deploy the model for scalable prediction
