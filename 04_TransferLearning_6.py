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
checkpointDir = saveModelDir + f"/checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    # create checkpoint callback
    ckptCallback = keras.callbacks.ModelCheckpoint(checkpointDir,
                                                   monitor="val_accuracy",
                                                   save_weights_only=True,
                                                   save_best_only=True,
                                                   verbose=1)
    # setup data augmentation
    dataAugmentation = keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomHeight(0.2),
        RandomWidth(0.2),
        RandomZoom(0.2),
        Rescaling(1 / 255.)
    ], name="DataAugment")

    # create model
    # baseModel = keras.applications.EfficientNetB0(include_top=False)
    baseModel: keras.Model = keras.models.load_model(preTrainedModelDir, compile=False)
    baseModel = keras.Model(inputs=baseModel.input, outputs=baseModel.layers[-2].output)
    baseModel.trainable = False
    # baseModel.summary()
    # setup model architecture with trainable top layers
    inputs = keras.layers.Input(shape=imgShape)
    x = dataAugmentation(inputs)
    x = baseModel(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(len(trainData.class_names),
                                 activation="softmax")(x)
    # initialize the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(keras.optimizers.Adam(),
                  keras.losses.CategoricalCrossentropy(),
                  ["accuracy"])

    model.summary()
    history = model.fit(trainData,
                        epochs=initialEpochs,
                        steps_per_epoch=len(trainData),
                        validation_data=testData,
                        validation_steps=int(0.15 * len(testData)),
                        callbacks=[ckptCallback])
    # start fine tuning
    baseModel.trainable = True
    for layer in baseModel.layers[:-10]:
        layer.trainable = False
    model.compile(keras.optimizers.Adam(1e-4),
                  keras.losses.CategoricalCrossentropy(),
                  ["accuracy"])
    model.summary()
    history_fineTuning = model.fit(trainData,
                                   epochs=initialEpochs + 5,
                                   steps_per_epoch=len(trainData),
                                   validation_data=testData,
                                   validation_steps=int(0.15 * len(testData)),
                                   callbacks=[ckptCallback],
                                   initial_epoch=history.epoch[-1])
    model.save(saveModelDir)
    model.evaluate(trainData)
    pass


if __name__ == "__main__":
    main()
