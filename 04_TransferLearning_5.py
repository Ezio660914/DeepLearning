# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 04_TransferLearning_5.py.py
@time: 2021-08-25 20:59
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
savedModelDir = "./savedModel/food101_MultiClass_3"
foodCategory = np.array(sorted(os.listdir(trainDir)))
imgShape = (224, 224, 3)
numClasses = len(foodCategory)
epochs = 5


def main():
    # PlotImageRandomly(trainDir, random.choice(foodCategory))
    # plt.show()
    # preprocess

    trainData = keras.preprocessing.image_dataset_from_directory(trainDir,
                                                                 image_size=imgShape[:2],
                                                                 label_mode="categorical",
                                                                 batch_size=32)

    testData = keras.preprocessing.image_dataset_from_directory(testDir,
                                                                image_size=imgShape[:2],
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
    baseModel = keras.applications.EfficientNetB1(include_top=False)
    # freeze the base model
    baseModel.trainable = False

    # create inputs into the model
    inputs = keras.layers.Input(shape=imgShape,
                                name="Inputlayer")

    # if using a model like ResNet50V2, need to normalize inputs
    # x=keras.layers.experimental.preprocessing.Rescaling(1/255.)(inputs)

    # add in data augmentation sequential model as a layer
    x = dataAugmentation(inputs)
    # pass the inputs to the base model
    x = baseModel(x, training=False)
    # average pool the outputs of the base model
    x = keras.layers.GlobalAvgPool2D(name="AveragePoolLayer")(x)
    # create the output activation layer
    outputs = keras.layers.Dense(10, activation="softmax", name="OutputLayer")(x)
    # combine the inputs with the outputs into a model

    model = keras.Model(inputs=inputs, outputs=outputs)
    # compile the model
    model.compile(keras.optimizers.Adam(learning_rate=1e-3),
                  keras.losses.CategoricalCrossentropy(),
                  ["accuracy"])
    model.summary()

    # create a model checkpoint callback
    # set checkpoint path
    checkpointDir = savedModelDir + f"/checkpoint/checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpointCallback = keras.callbacks.ModelCheckpoint(checkpointDir,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         save_freq="epoch",
                                                         verbose=1)

    # fit the model
    history = model.fit(trainData,
                        epochs=epochs,
                        steps_per_epoch=len(trainData),
                        validation_data=testData,
                        validation_steps=len(testData),
                        validation_freq=1,
                        callbacks=[checkpointCallback],
                        )

    # start fine-tuning
    baseModel.trainable = True
    # fine-tuning with the last 10 layers
    for layer in baseModel.layers[:-10]:
        layer.trainable = False
    for layer in baseModel.layers:
        print((layer.name, layer.trainable))
    model.compile(keras.optimizers.Adam(learning_rate=1e-4),  # learning_rate is 10x lower than before for fine-tuning
                  keras.losses.CategoricalCrossentropy(),
                  ["accuracy"])
    model.summary()
    history_fineTuning = model.fit(trainData,
                                   epochs=epochs + 5,  # Fine tune for another 5 epochs
                                   steps_per_epoch=len(trainData),
                                   validation_data=testData,
                                   validation_steps=len(testData),
                                   validation_freq=1,
                                   callbacks=[checkpointCallback],
                                   initial_epoch=history.epoch[-1]  # start from previous last epoch
                                   )
    model.save(savedModelDir)
    pass


if __name__ == "__main__":
    main()
