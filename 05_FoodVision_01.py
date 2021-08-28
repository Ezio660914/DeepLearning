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
initialEpochs = 3
numClasses = len(os.listdir(trainDir))


def main():
    # obtain train dataset
    trainData = keras.preprocessing.image_dataset_from_directory(trainDir,
                                                                 label_mode="categorical",
                                                                 batch_size=32,
                                                                 image_size=imgSize,
                                                                 shuffle=True)
    print([i for i in trainData.take(1)])
    # obtain test dataset
    testData = keras.preprocessing.image_dataset_from_directory(testDir,
                                                                label_mode="categorical",
                                                                batch_size=32,
                                                                image_size=imgSize,
                                                                shuffle=False)

    # create model callbacks
    ckptCallback = keras.callbacks.ModelCheckpoint(checkpointDir,
                                                   monitor="val_accuracy",
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   verbose=1)
    # setup mixed precision training
    keras.mixed_precision.set_global_policy("mixed_float16")
    # build feature extraction model
    baseModel = keras.applications.EfficientNetB0(include_top=False)
    baseModel.trainable = False
    # create functional model
    inputs = keras.layers.Input(imgShape)
    x = baseModel(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(numClasses)(x)
    outputs = keras.layers.Activation("softmax", dtype=tf.float32)(x)
    model = keras.Model(inputs, outputs)
    # compile the model
    model.compile(keras.optimizers.Adam(),
                  keras.losses.CategoricalCrossentropy(),
                  ["accuracy"])
    model.summary()
    # for layer in model.layers:
    #     print((layer.name, layer.trainable, layer.dtype, layer.dtype_policy))

    # fit the model
    history = model.fit(trainData,
                        epochs=initialEpochs,
                        steps_per_epoch=len(trainData),
                        validation_data=testData,
                        validation_steps=int(0.15 * len(testData)),
                        callbacks=[ckptCallback])
    model.evaluate(testData)
    model.save(saveModelDir)
    pass


if __name__ == "__main__":
    main()
