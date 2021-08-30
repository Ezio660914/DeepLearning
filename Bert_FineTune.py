# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: Bert_FineTune.py
@time: 2021-08-30 11:58
"""

import os
import pathlib
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from official.modeling import tf_utils
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

bertDir = "savedModel/small_bert_bert_en_uncased_L-8_H-512_A-8_2"
resourceDir = "resource/MSRC"


def EncodeSentence(sentence, tokenizer):
    tokens = list(tokenizer.tokenize(sentence.numpy()))
    tokens.append("[SEP]")
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids


def BertEncode(glueDict, tokenizer):
    numExamples = len(glueDict["sentence1"])
    sentence1 = tf.ragged.constant([EncodeSentence(s, tokenizer) for s in glueDict["sentence1"]])
    sentence2 = tf.ragged.constant([EncodeSentence(s, tokenizer) for s in glueDict["sentence2"]])
    cls = [tokenizer.convert_tokens_to_ids(["[CLS]"])] * numExamples
    inputWordIds = tf.concat([cls, sentence1, sentence2], axis=-1)
    inputMask = tf.ones_like(inputWordIds).to_tensor()
    typeCls = tf.zeros_like(cls)
    typeS1 = tf.zeros_like(sentence1)
    typeS2 = tf.ones_like(sentence2)
    inputTypeIds = tf.concat([typeCls, typeS1, typeS2], axis=-1).to_tensor()
    bertInputs = dict(
        input_word_ids=inputWordIds.to_tensor(),
        input_mask=inputMask,
        input_type_ids=inputTypeIds)
    return bertInputs


def BuildModel():
    # config bert model
    input1 = keras.layers.Input(shape=(None,), name="input_word_ids", dtype=tf.int32)
    input2 = keras.layers.Input(shape=(None,), name="input_mask", dtype=tf.int32)
    input3 = keras.layers.Input(shape=(None,), name="input_type_ids", dtype=tf.int32)
    bertModel = hub.KerasLayer(bertDir, trainable=True)
    bertInputArgs = {
        'input_word_ids': input1,
        'input_mask': input2,
        'input_type_ids': input3,
    }
    bertOutput = bertModel(bertInputArgs, False)
    net = bertOutput["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1)(net)
    net = tf.keras.layers.Activation(tf.keras.activations.sigmoid, dtype=tf.float32)(net)
    model = tf.keras.Model([input1, input2, input3], net)
    return model


def main():
    # get the data
    glue, info = tfds.load("glue/mrpc",
                           with_info=True,
                           batch_size=-1,
                           data_dir=resourceDir)
    # preprocess the data
    # use bert tokenizer
    tokenizer = bert.tokenization.FullTokenizer(vocab_file=bertDir + "/assets/vocab.txt",
                                                do_lower_case=True)
    # encode data
    glueTrain = BertEncode(glue["train"], tokenizer)
    glueTrainLabels = glue["train"]["label"]
    glueVal = BertEncode(glue["validation"], tokenizer)
    glueValLabels = glue["validation"]["label"]
    glueTest = BertEncode(glue["test"], tokenizer)
    glueTestLabels = glue["test"]["label"]
    model = BuildModel()

    pass


if __name__ == "__main__":
    main()
