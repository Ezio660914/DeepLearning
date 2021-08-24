# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 03_CNN_1.py
@time: 2021-08-22 10:32
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

trainDataDir = "./resource/food101/pizza_steak/train/"
testDataDir = "./resource/food101/pizza_steak/test/"
foodCategory = ["pizza", "steak"]


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
    # img = PlotImageRandomly(trainDataDir, foodCategory[0])
    tf.random.set_seed(32)

    # preprocess data, normalization
    # use ImageDataGenerator to do data augmentation
    trainDataGen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=0.1,
                                      shear_range=0.1,
                                      zoom_range=0.1,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      horizontal_flip=True)
    validDataGen = ImageDataGenerator(rescale=1. / 255)

    # import data from directories and turn it into batches
    trainData = trainDataGen.flow_from_directory(trainDataDir,
                                                 batch_size=32,
                                                 target_size=(224, 224),
                                                 class_mode="binary",
                                                 seed=43)
    testData = validDataGen.flow_from_directory(testDataDir,
                                                batch_size=32,
                                                target_size=(224, 224),
                                                class_mode="binary",
                                                seed=43)
    # visualise the image
    images, labels = trainData.next()
    i = random.randint(0, len(images) - 1)
    plt.imshow(images[i])
    plt.title(labels[i])
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.xlabel(images[i].shape)
    plt.show()
    # create cnn model (use tiny VGG model)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=10,
                               kernel_size=3,
                               activation=tf.keras.activations.relu,
                               input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(10, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(10, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(10, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, tf.keras.activations.relu),
        tf.keras.layers.Dense(10, tf.keras.activations.relu),
        tf.keras.layers.Dense(1, tf.keras.activations.sigmoid)
    ])
    # define a callback
    callback = tf.keras.callbacks.EarlyStopping("val_binary_accuracy", patience=3)

    model.compile(tf.keras.optimizers.Adam(),
                  tf.keras.losses.BinaryCrossentropy(),
                  [tf.keras.metrics.BinaryAccuracy()])
    model.summary()
    history = model.fit(trainData,
                        epochs=5,
                        steps_per_epoch=len(trainData),
                        validation_data=testData,
                        validation_steps=len(testData),
                        callbacks=[callback])
    # save the model
    model.save("./savedModel/food101/")
    pd.DataFrame(history.history).plot()
    plt.show()
    pass


if __name__ == "__main__":
    main()
