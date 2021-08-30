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


class StaticConst:
    bertDir = "savedModel/small_bert_bert_en_uncased_L-8_H-512_A-8_2"
    resourceDir = "resource/MSRC"
    pass


def main():
    # get the data
    glue, info = tfds.load("glue/mrpc",
                           with_info=True,
                           batch_size=-1,
                           data_dir=StaticConst.resourceDir)
    # get the train data
    trainDict = glue["train"]
    # preprocess the data
    # use bert tokenizer
    tokenizer = bert.tokenization.FullTokenizer(vocab_file=StaticConst.bertDir + "/assets/vocab.txt",
                                                do_lower_case=True)
    
    pass


if __name__ == "__main__":
    main()
