import logging
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from transformers import RobertaModel
from argparse import ArgumentParser
import multiprocessing as mp


class TextClassifier(pl.LightningModule):
    def __init__(
        self, 
        num_classes: int, 
        model_name: str,
        learning_rate: float,
        batch_size: int,
        pretrained_model_path: str,
        **kwargs
    ):
        super(TextClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.transformer = RobertaModel.from_pretrained(pretrained_model_path, from_tf=False)
        self.classifier = torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=4)
        parser.add_argument('--model_name', type=str, default='roberta-large')
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--pretrained_model_path', type=str, default='roberta-large')
        return parser
    
    def get_dataloader(
        self, 
        dataset,
        shuffle: bool,
        pin_memory: bool,
    ):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=mp.cpu_count(),
            pin_memory=pin_memory,
        )
    
    def forward(self, input_ids, attention_mask):
        # in lightning, forward defines the prediction/inference actions
        base_model_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = base_model_output.pooler_output
        logging.debug("pooler output shape={}".format(pooler_output.shape))
        output = self.classifier(pooler_output)
        logging.debug("output shape={}".format(output.shape))
        return torch.argmax(output, dim=1)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x = {k: v for k, v in batch.items() if k != 'labels'}
        y = batch['labels']
        base_model_output = self.transformer(
            input_ids=x['input_ids'], 
            attention_mask=x['attention_mask']
        )
        pooler_output = base_model_output.pooler_output
        logits = self.classifier(pooler_output)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x = {k: v for k, v in batch.items() if k != 'labels'}
        y = batch['labels']

        base_model_output = self.transformer(
            input_ids=x['input_ids'], 
            attention_mask=x['attention_mask']
        )
        pooler_output = base_model_output.pooler_output
        logits = self.classifier(pooler_output)
        loss = self.loss_fn(logits, y)
        self.valid_acc(logits, y)
        self.log('valid_acc', self.valid_acc, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
       x = {k: v for k, v in batch.items() if k != 'labels'}
       y = batch['labels']

       base_model_output = self.transformer(
           input_ids=x['input_ids'],
           attention_mask=x['attention_mask']
       )
       pooler_output = base_model_output.pooler_output
       logits = self.classifier(pooler_output)
       loss = self.loss_fn(logits, y)
       self.test_acc(logits, y)
       self.log('test_acc', self.valid_acc, on_step=True,
                on_epoch=True, prog_bar=True, logger=True)
       self.log('test_loss', loss, on_step=True,
                on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
