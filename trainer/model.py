import logging
import torch
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from transformers import RobertaModel
from argparse import ArgumentParser

class TextClassifier(pl.LightningModule):
    def __init__(
        self, 
        num_classes: int, 
        model_name: str,
        learning_rate: float,
        **kwargs
    ):
        super(TextClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.loss_fn = CrossEntropyLoss()
        self.transformer = RobertaModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_classes', type=int, default=4)
        parser.add_argument('--model_name', type=str, default='roberta-large')
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser
    
    def forward(self, input_ids, attention_mask):
        # in lightning, forward defines the prediction/inference actions
        base_model_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = base_model_output.pooler_output
        logging.debug("pooler output shape={}".format(pooler_output.shape))
        output = self.classifier(pooler_output)
        logging.debug("output shape={}".format(output.shape))
        return output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x = {k: v for k, v in batch.items() if k != 'labels'}
        y = batch['labels']

        base_model_output = self.transformer(
            input_ids=x['input_ids'], 
            attention_mask=x['attention_mask']
        )
        pooler_output = base_model_output.pooler_output
        logging.debug("pooler output shape={}".format(pooler_output.shape))
        predictions = self.classifier(pooler_output)
        logging.debug("output shape={}".format(predictions.shape))

        loss = self.loss_fn(predictions, y)
        self.log('train_loss', loss)
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
        predictions = self.classifier(pooler_output)
        loss = self.loss_fn(predictions, y)
        self.log('validation_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
