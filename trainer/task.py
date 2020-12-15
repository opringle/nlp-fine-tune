import argparse
import logging

from datasets import load_dataset
from transformers import RobertaTokenizer

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from .model import TorchTextClassifier

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
            'MultiWorkerMirroredStrategy'
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
        default=False,
        help="""
        Flag to decide if the model checkpoint should be
        re-used from the job-dir.
        If set to False then the job-dir will be deleted.
        """
    )
    return parser.parse_args()


def get_dataloader(dataset, tokenizer, batch_size=32):
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    
    MODEL_NAME = 'roberta-large'
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # load, encode and format input data for pytorch model
    datasets = load_dataset('ag_news')
    train_ds = datasets['train']
    test_ds = datasets['test']
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_dataloader = get_dataloader(train_ds, tokenizer, batch_size=BATCH_SIZE)
    test_dataloader = get_dataloader(test_ds, tokenizer, batch_size=BATCH_SIZE)


    model = TorchTextClassifier(num_classes=4)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    loss_fn = CrossEntropyLoss()

    # fit the model to the data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)
    for epoch in range(EPOCHS):
        for i, batch in enumerate(tqdm(train_dataloader)):
            features = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            predictions = model(**features)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                logging.info("Epoch {} batch {} loss={:.3f}".format(epoch, i, loss))
