import logging
import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFRobertaModel

class KerasTextClassifier(tf.keras.Model):
    def __init__(self, num_classes: int, pretrained_roberta_name: str):
        super(KerasTextClassifier, self).__init__()
        self.transformer = TFRobertaModel.from_pretrained(pretrained_roberta_name)
        self.final_layer = layers.Dense(num_classes)

    def call(self, inputs, training=False, mask=None):
        transformer_features = self.transformer(inputs)
        print(transformer_features)
        # TODO: clearly fooked
        output = self.final_layer(transformer_features)
        return output
