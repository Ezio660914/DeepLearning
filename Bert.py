# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: Bert.py
@time: 2021-08-29 13:08
"""

import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pathlib
import datetime

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")


class StaticConst:
    trainDir = pathlib.Path("resource/aclImdb/train")
    testDir = pathlib.Path("resource/aclImdb/test")
    batchSize = 32
    seed = 42
    bertModelDir = pathlib.Path("savedModel/small_bert_bert_en_uncased_L-8_H-512_A-8_2")
    bertPreprocessDir = pathlib.Path("savedModel/bert_en_uncased_preprocess_3")


def main():
    # get the train dataset
    trainDs = keras.preprocessing.text_dataset_from_directory(str(StaticConst.trainDir),
                                                              batch_size=StaticConst.batchSize,
                                                              validation_split=0.2,
                                                              subset="training",
                                                              seed=StaticConst.seed)
    classNames = trainDs.class_names
    trainDs = trainDs.cache().prefetch(tf.data.AUTOTUNE)

    # get a validation dataset
    valDs = keras.preprocessing.text_dataset_from_directory(str(StaticConst.trainDir),
                                                            batch_size=StaticConst.batchSize,
                                                            validation_split=0.2,
                                                            subset="validation",
                                                            seed=StaticConst.seed)
    valDs = valDs.cache().prefetch(tf.data.AUTOTUNE)

    # get the test dataset
    testDs = keras.preprocessing.text_dataset_from_directory(str(StaticConst.testDir),
                                                             batch_size=StaticConst.batchSize)
    testDs = testDs.cache().prefetch(tf.data.AUTOTUNE)

    # view some train examples
    print("Training Examples:")
    for textBatch, labelBatch in trainDs.take(1):
        for i in range(3):
            print(f"Review: {textBatch.numpy()[i]}")
            label = labelBatch.numpy()[i]
            print(f"Label: {label} ({classNames[label]})")
    print("\n")

    # try the preprocess model on some text
    bertPreprocessModel = hub.KerasLayer(str(StaticConst.bertPreprocessDir))
    textTest = ["this is such an amazing movie!"]
    textTestPreprocessed = bertPreprocessModel(textTest)
    print(f"Keys\t: {list(textTestPreprocessed.keys())}")
    print(f"Shape\t: {textTestPreprocessed['input_word_ids'].shape}")
    print(f"Word Ids\t: {textTestPreprocessed['input_word_ids'][0, :12]}")
    print(f"Input Mask\t: {textTestPreprocessed['input_mask'][0, :12]}")
    print(f"Type Ids\t: {textTestPreprocessed['input_type_ids'][0, :12]}")
    pass


if __name__ == "__main__":
    main()
