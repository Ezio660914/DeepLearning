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

foodCategory = ["pizza", "steak"]
imgPath = "resource/customImage/steak2.jpeg"


def LoadImage(file, shape=None):
    if shape is None:
        shape = [224, 224]
    # read in the image
    img = tf.io.read_file(file)
    # decode the read file into a tensor
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, shape)
    # rescale the value
    img = img / 255.0
    return img


def Predict(model):
    img = LoadImage(imgPath)
    imgRaw = mpimg.imread(imgPath)
    print(img.shape)
    pred = model.predict(tf.expand_dims(img, 0))
    predClass = foodCategory[int(tf.round(pred))]
    print(predClass)
    plt.imshow(imgRaw)
    plt.title(f"Probability:{tf.squeeze(pred)}\nPrediction: {predClass}")
    plt.axis(False)


def main():
    model = tf.keras.models.load_model("./savedModel/food101")
    model.summary()
    Predict(model)
    plt.show()
    pass


if __name__ == "__main__":
    main()
