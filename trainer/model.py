import logging
import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFRobertaModel

class KerasTextClassifier(tf.keras.Model):
    def __init__(self, num_classes: int, pretrained_roberta_name: str):
        super(KerasTextClassifier, self).__init__()
        self.transformer = TFRobertaModel.from_pretrained(pretrained_roberta_name)
        self.final_layer = layers.Dense(num_classes)
        self.softmax = layers.Softmax()

    def call(self, inputs, training=False, mask=None):
        transformer_features = self.transformer(inputs)
        pooler_output = transformer_features.pooler_output
        logging.debug("pooler output shape={}".format(pooler_output.shape))
        output = self.softmax(self.final_layer(pooler_output))
        logging.debug("output shape={}".format(output.shape))
        return output
