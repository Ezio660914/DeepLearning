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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
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
trainDir = "resource/food101/10_food_classes_all_data/train/"
testDir = "resource/food101/10_food_classes_all_data/test/"
foodCategory = np.array(sorted(os.listdir(trainDir)))
imgShape = (224, 224, 3)
numClasses = len(foodCategory)
savedModelDir = [
    "savedModel/efficientnet_b0_feature-vector_1",
    "savedModel/imagenet_resnet_v2_50_feature_vector_5"
]


def PlotImageRandomly(imageDir, category):
    imageDir = imageDir + category + "/"
    randomImage = random.sample(os.listdir(imageDir), 1)
    img = mpimg.imread(imageDir + randomImage[0])
    plt.imshow(img)
    plt.title(category)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.xlabel(img.shape)
    return img


def CreateTensorboardCallback(dir, experimentName):
    logDir = dir + "/" + experimentName + "/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tbCallback = tf.keras.callbacks.TensorBoard(logDir)
    print(f"Saving TensorBoard log files to: {logDir}")
    return tbCallback


def CreateModel(modelDir, nClasses):
    featureExtractorLayer = hub.KerasLayer(modelDir,
                                           trainable=False,
                                           input_shape=imgShape)  # freeze the already learned patterns
    model = tf.keras.Sequential([
        featureExtractorLayer,
        tf.keras.layers.Dense(nClasses, "softmax")
    ])
    model.compile(tf.keras.optimizers.Adam(),
                  tf.keras.losses.CategoricalCrossentropy(),
                  [tf.keras.metrics.CategoricalAccuracy()])
    return model


def main():
    # PlotImageRandomly(trainDir, random.choice(foodCategory))
    # plt.show()
    # preprocess
    # trainDataGen = ImageDataGenerator(rescale=1 / 255.,
    #                                   rotation_range=0.1,
    #                                   shear_range=0.1,
    #                                   zoom_range=0.1,
    #                                   width_shift_range=0.1,
    #                                   height_shift_range=0.1,
    #                                   horizontal_flip=True)
    trainDataGen = ImageDataGenerator(rescale=1 / 255.)
    testDataGen = ImageDataGenerator(rescale=1 / 255.)
    # load data
    trainData = trainDataGen.flow_from_directory(trainDir,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode="categorical")
    testData = testDataGen.flow_from_directory(testDir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode="categorical")

    # set up callbacks
    # track experiment with tensorboard callback
    callback = CreateTensorboardCallback("./log/tensorflow_hub", "resNet50V2")
    # create a model with tensorflow hub
    resNetModel = CreateModel(savedModelDir[1], numClasses)
    resNetModel.summary()

    resNetHistory = resNetModel.fit(trainData,
                                    epochs=5,
                                    steps_per_epoch=len(trainData),
                                    validation_data=testData,
                                    validation_steps=len(testData),
                                    callbacks=[callback])
    pass


if __name__ == "__main__":
    main()
