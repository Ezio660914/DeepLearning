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
    pass


if __name__ == "__main__":
    main()
