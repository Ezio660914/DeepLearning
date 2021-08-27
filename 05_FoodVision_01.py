# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 05_FoodVision_01.py
@time: 2021-08-27 9:29
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
import pathlib
import datetime

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")

# setup the train and test directories
trainDir = "./resource/food101/food-101/train"
testDir = "./resource/food101/food-101/test"
logDir = "./log/tensorflow_hub"
saveModelDir = "./savedModel/FoodVision_1"
preTrainedModelDir = "./savedModel/efficientnet_b0_feature-vector_1"
checkpointDir = saveModelDir + f"/checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
imgShape = (224, 224, 3)
imgSize = (224, 224)
initialEpochs = 5


def PreProcessImg(imagePath, imgSize=(224, 224)):
    """
    Converts image datatype from uint8 to float32,
    re-range from 255 to 1, and reshapes image to the given imgSize
    """
    file = tf.io.read_file(imagePath)
    image = tf.image.decode_image(file, channels=3, expand_animations=False)
    image = tf.image.resize(image, imgSize)
    image = image / 255.0
    return image


def GetDataPathList(dataDir):
    dataPathList = list(pathlib.Path(dataDir).glob("*/*"))
    return [str(path) for path in dataPathList]


def GetLabelList(dataDir):
    from sklearn.preprocessing import OneHotEncoder
    dataPathList = list(pathlib.Path(dataDir).glob("*/*"))
    classNames = np.array([path.parent.name for path in dataPathList])[:, np.newaxis]
    classNamesOneHot = OneHotEncoder().fit_transform(classNames).toarray().astype(np.int64)
    return classNamesOneHot


def main():
    # obtain the train dataset
    trainPathDs = tf.data.Dataset.from_tensor_slices(GetDataPathList(trainDir))
    # load and preprocess the train image as dataset
    trainImgDs = trainPathDs.map(map_func=PreProcessImg, num_parallel_calls=tf.data.AUTOTUNE)
    # load the labels
    trainLabelDs = tf.data.Dataset.from_tensor_slices(GetLabelList(trainDir))
    # zip as (image,label) dataset
    trainImgLabelDs = tf.data.Dataset.zip((trainImgDs, trainLabelDs))
    # shuffle train data and turn it into batches and prefetch it (load it faster)
    trainData = trainImgLabelDs.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    pass


if __name__ == "__main__":
    main()
