# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 04_TransferLearning_6.py.py
@time: 2021-08-26 11:34
"""

import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import datetime

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")

# setup the train and test directories
trainDir = "resource/food101/101_food_classes_10_percent/train/"
testDir = "resource/food101/101_food_classes_10_percent/test/"
logDir = "./log/tensorflow_hub"
saveModelDir = "./savedModel/food101_101Classes_1"
preTrainedModelDir = "./savedModel/efficientnet_b0_feature-vector_1"
imgShape = (224, 224, 3)
imgSize = (224, 224)
initialEpochs = 5


def main():
    trainData = keras.preprocessing.image_dataset_from_directory(trainDir,
                                                                 label_mode="categorical",
                                                                 batch_size=32,
                                                                 image_size=imgSize)
    testData = keras.preprocessing.image_dataset_from_directory(testDir,
                                                                label_mode="categorical",
                                                                batch_size=32,
                                                                image_size=imgSize,
                                                                shuffle=False)
    
    pass


if __name__ == "__main__":
    main()
