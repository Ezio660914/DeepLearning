# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 04_TransferLearning_1.py
@time: 2021-08-25 11:01
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
trainDir = "resource/food101/10_food_classes_10_percent/train/"
testDir = "resource/food101/10_food_classes_all_data/test/"
logDir = "./log/tensorflow_hub"
foodCategory = np.array(sorted(os.listdir(trainDir)))
imgShape = (224, 224, 3)
numClasses = len(foodCategory)
epochs = 5


def main():
    # PlotImageRandomly(trainDir, random.choice(foodCategory))
    # plt.show()
    # preprocess

    trainData = keras.preprocessing.image_dataset_from_directory(trainDir,
                                                                 image_size=(224, 224),
                                                                 label_mode="categorical",
                                                                 batch_size=32)

    testData = keras.preprocessing.image_dataset_from_directory(testDir,
                                                                image_size=(224, 224),
                                                                label_mode="categorical",
                                                                batch_size=32)

    # adding data augmentation
    dataAugmentation = keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomHeight(0.1),
        RandomWidth(0.1)
    ])

    # visualise data augmentation layer
    targetClass = random.choice(trainData.class_names)
    targetDir = trainDir + "/" + targetClass
    randomImg = random.choice(os.listdir(targetDir))
    randomImgDir = targetDir + "/" + randomImg
    img = mpimg.imread(randomImgDir)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"{targetClass} (Original)")
    plt.axis(False)

    augmentedImg = dataAugmentation(tf.expand_dims(img, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(augmentedImg) / 255.)
    plt.title(f"{targetClass} (Augmented)")
    plt.axis(False)

    plt.show()
    # create the model with keras functional api
    # create base model with tf.keras.applications
    baseModel = keras.applications.EfficientNetB0(False)
    # freeze the base model
    baseModel.trainable = False
    # create inputs into the model
    inputs = keras.layers.Input(shape=imgShape,
                                name="Inputlayer")
    # if using a model like ResNet50V2, need to normalize inputs
    # x=keras.layers.experimental.preprocessing.Rescaling(1/255.)(inputs)
    # pass the inputs to the base model
    x = baseModel(inputs)
    print(x.shape)
    # average pool the outputs of the base model
    x = keras.layers.GlobalAvgPool2D(name="AveragePoolLayer")(x)
    print(x.shape)
    # create the output activation layer
    outputs = keras.layers.Dense(10, activation="softmax", name="OutputLayer")(x)
    # combine the inputs with the outputs into a model
    model = keras.Model(inputs, outputs)
    # compile the model
    model.compile(keras.optimizers.Adam(),
                  keras.losses.CategoricalCrossentropy(),
                  ["accuracy"])
    model.summary()
    # fit the model
    history = model.fit(trainData,
                        epochs=epochs,
                        steps_per_epoch=len(trainData),
                        validation_data=testData,
                        validation_steps=len(testData),
                        validation_freq=1)
    pass


if __name__ == "__main__":
    main()
