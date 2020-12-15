import logging
import torch
from transformers import RobertaModel

class TorchTextClassifier(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(TorchTextClassifier, self).__init__()
        self.transformer = RobertaModel.from_pretrained("roberta-large")
        self.classifier = torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, input_ids, attention_mask):
        base_model_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = base_model_output.pooler_output
        logging.debug("pooler output shape={}".format(pooler_output.shape))
        output = self.classifier(pooler_output)
        logging.debug("output shape={}".format(output.shape))
        return output
