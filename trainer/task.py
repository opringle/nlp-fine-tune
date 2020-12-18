import argparse
import logging
import sys
import os
from datasets import load_dataset
from transformers import RobertaTokenizer
import hypertune
import json
import time
import torch
import pytorch_lightning as pl
import multiprocessing as mp

from .model import TextClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='GCP training application')
    
    parser.add_argument('--job-dir', required=True, type=str)
    parser.add_argument('--train-data-file', required=True)
    parser.add_argument('--batch-size', type=int, required=True)

    # add model specific args (OOOOH)
    parser = TextClassifier.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    return parser.parse_args()


def get_dataloader(dataset, tokenizer, batch_size=32, shuffle=True):
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = torch.utils.data.DataLoader(
      dataset, 
      batch_size=batch_size, 
      shuffle=shuffle,
      num_workers=mp.cpu_count()
    )
    return dataloader


def wait_for_tpu_cluster_resolver_ready():
  """Waits for `TPUClusterResolver` to be ready and return it.

  Returns:
    A TPUClusterResolver if there is TPU machine (in TPU_CONFIG). Otherwise,
    return None.
  Raises:
    RuntimeError: if failed to schedule TPU.
  """
  tpu_config_env = os.environ.get('TPU_CONFIG')
  if not tpu_config_env:
    logging.info('Missing TPU_CONFIG, use CPU/GPU for training.')
    return None

  tpu_node = json.loads(tpu_config_env)
  logging.info('Waiting for TPU to be ready: \n%s.', tpu_node)

  num_retries = 40
  for i in range(num_retries):
    try:
      tpu_cluster_resolver = (
          tf.distribute.cluster_resolver.TPUClusterResolver(
              tpu=[tpu_node['tpu_node_name']],
              zone=tpu_node['zone'],
              project=tpu_node['project'],
              job_name='worker'))
      tpu_cluster_resolver_dict = tpu_cluster_resolver.cluster_spec().as_dict()
      if 'worker' in tpu_cluster_resolver_dict:
        logging.info('Found TPU worker: %s', tpu_cluster_resolver_dict)
        return tpu_cluster_resolver
    except Exception as e:
      if i < num_retries - 1:
        logging.info('Still waiting for provisioning of TPU VM instance.')
      else:
        # Preserves the traceback.
        raise RuntimeError('Failed to schedule TPU: {}'.format(e))
    time.sleep(10)

  # Raise error when failed to get TPUClusterResolver after retry.
  raise RuntimeError('Failed to schedule TPU.')


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    MODEL_NAME = 'roberta-large'
    BATCH_SIZE = args.batch_size

    # get loaders for model
    datasets = load_dataset('ag_news')
    train_ds = datasets['train']
    test_ds = datasets['test']
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_dataloader = get_dataloader(train_ds, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = get_dataloader(test_ds, tokenizer, batch_size=BATCH_SIZE, shuffle=False)

    model = TextClassifier(**vars(args))  # TODO: fix
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(
      model,
      train_dataloader,
      test_dataloader,
    )
    
    # loss, accuracy = model.evaluate(test_ds)
    # logging.info("Accuracy {}".format(accuracy))

    # hpt = hypertune.HyperTune()
    # hpt.report_hyperparameter_tuning_metric(
    #     hyperparameter_metric_tag='accuracy',
    #     metric_value=accuracy,
    #     # global_step=1000
    # )
