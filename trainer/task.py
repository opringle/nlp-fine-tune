import argparse
import logging
import sys
import os
from datasets import load_dataset
from transformers import RobertaTokenizer
import hypertune
import tensorflow as tf
from tensorflow import keras
import json
import time

from .model import KerasTextClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='GCP training application')
    
    group = parser.add_argument_group('data')
    group.add_argument('--train-data-file', required=True)

    group = parser.add_argument_group('hyperparameters')
    group.add_argument('--epochs', type=int, required=True)
    group.add_argument('--batch-size', type=int, required=True)

    group = parser.add_argument_group('compute')
    group.add_argument(
        '--distribution-strategy', 
        type=str, 
        default=None,
        choices=[
            'MirroredStrategy',
            'MultiWorkerMirroredStrategy',
            'tpu',
        ],
    )
    
    group = parser.add_argument_group('artifacts')
    group.add_argument(
        '--job-dir',
        required=True,
        type=str,
        help='GCS location to write checkpoints and export models.',
    )
    group.add_argument(
        '--reuse-job-dir',
        action='store_true',
        help="""
        Flag to decide if the model checkpoint should be
        re-used from the job-dir.
        If set to False then the job-dir will be deleted.
        """
    )
    return parser.parse_args()


def get_ds(dataset, tokenizer, batch_size=32, shuffle=True):
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'labels'])
    features = {x: dataset[x].to_tensor(default_value=0, shape=[None, 512]) for x in ['input_ids', 'attention_mask']}
    ds = tf.data.Dataset.from_tensor_slices((features, dataset["labels"]))
    if shuffle:
        ds.shuffle(buffer_size=len(dataset))
    ds = ds.batch(batch_size)
    return ds


def get_strategy(distribution_strategy: str):
    if distribution_strategy == 'MirroredStrategy':
        strategy = tf.distribute.MirroredStrategy()
    elif distribution_strategy == 'MultiWorkerMirroredStrategy':
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    elif distribution_strategy == 'tpu':
        resolver = wait_for_tpu_cluster_resolver_ready()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    logging.info('Training with strategy: {}'.format(strategy))
    return strategy


def _setup_logging():
  """Sets up logging."""
  root_logger = logging.getLogger()
  root_logger_previous_handlers = list(root_logger.handlers)
  for h in root_logger_previous_handlers:
    root_logger.removeHandler(h)
  root_logger.setLevel(logging.INFO)
  root_logger.propagate = False

  # Set tf logging to avoid duplicate logging. If the handlers are not removed,
  # then we will have duplicate logging
  tf_logger = logging.getLogger('TensorFlow')
  while tf_logger.handlers:
    tf_logger.removeHandler(tf_logger.handlers[0])

  # Redirect INFO logs to stdout
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.INFO)
  root_logger.addHandler(stdout_handler)

  # Suppress C++ level warnings.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    _setup_logging()
    
    MODEL_NAME = 'roberta-large'
    EPOCHS = args.epochs
    BATCH_SIZE_PER_REPLICA = args.batch_size

    # handle whether training on cpu, gpu, multi gpu, multi node multi gpu or tpu
    strategy = get_strategy(args.distribution_strategy)
    num_replicas_in_sync = strategy.num_replicas_in_sync
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_replicas_in_sync
    logging.info(
        'Number of devices: {}\t Batch size (before distribution): {}'.format(
            num_replicas_in_sync,
            BATCH_SIZE
        )
    )

    # get loaders for model
    datasets = load_dataset('ag_news')
    train_ds = datasets['train']
    test_ds = datasets['test']
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_ds = get_ds(train_ds, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
    test_ds = get_ds(test_ds, tokenizer, batch_size=BATCH_SIZE)

    # using scope() and fit() allows Keras to handle complexity of distributed training
    with strategy.scope():
        model = KerasTextClassifier(num_classes=4, pretrained_roberta_name=MODEL_NAME)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.job_dir, 'logs'))
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback],
    )
    
    loss, accuracy = model.evaluate(test_ds)
    logging.info("Accuracy {}".format(accuracy))

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=accuracy,
        # global_step=1000
    )
