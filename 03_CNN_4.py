# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 03_CNN_4.py
@time: 2021-08-24 21:24
"""

import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")

# setup the train and test directories
trainDir = "resource/food101/10_food_classes_all_data/train/"
testDir = "resource/food101/10_food_classes_all_data/test/"
foodCategory = np.array(sorted(os.listdir(trainDir)))


def PlotImageRandomly(imageDir, category):
    imageDir = imageDir + category + "/"
    randomImage = random.sample(os.listdir(imageDir), 1)
    img = mpimg.imread(imageDir + randomImage[0])
    plt.imshow(img)
    plt.title(category)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.xlabel(img.shape)
    return img


def main():
    PlotImageRandomly(trainDir, random.choice(foodCategory))
    # plt.show()
    # preprocess
    trainDataGen = ImageDataGenerator(rescale=1 / 255.)
    testDataGen = ImageDataGenerator(rescale=1 / 255.)
    # load data
    trainData = trainDataGen.flow_from_directory(trainDir,
                                                 target_size=(224, 224),
                                                 batch_size=32)
    testData = testDataGen.flow_from_directory(testDir,
                                               target_size=(224, 224),
                                               batch_size=32)
    # create a model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, 3,
                               activation="relu",
                               input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    model.compile(tf.keras.optimizers.Adam(),
                  tf.keras.losses.CategoricalCrossentropy(),
                  [tf.keras.metrics.CategoricalAccuracy()])
    model.summary()
    callback = tf.keras.callbacks.EarlyStopping(patience=3)
    model.fit(trainData,
              epochs=5,
              callbacks=callback,
              steps_per_epoch=len(trainData),
              validation_data=testData,
              validation_steps=len(testData))
    pass


if __name__ == "__main__":
    main()
