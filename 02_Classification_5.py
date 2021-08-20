# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 02_Classification_5.py
@time: 2021-08-20 11:20
"""
"""working with larger examples
multiclass classification"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools

# the data has already been sorted into training and test sets
"""Fashion-MNIST is a dataset of Zalando's article images
consisting of a training set of 60,000 examples and a test set of 10,000 examples.
Each example is a 28x28 grayscale image, associated with a label from 10 classes."""
(trainData, trainLabels), (testData, testLabels) = fashion_mnist.load_data()
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
              'Ankle boot']


def PlotExample(index):
    plt.imshow(trainData[index], cmap=plt.cm.get_cmap('binary'))
    plt.title(classNames[trainLabels[index]])
    plt.show()


def PlotExamplesRandomly():
    import random
    plt.figure(figsize=(7, 7))
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        index = random.randint(0, len(trainData) - 1)
        plt.imshow(trainData[index], cmap=plt.cm.get_cmap('binary'))
        plt.title(classNames[trainLabels[index]])
        plt.axis(False)
    plt.show()


def NeuralNetwork(X_train, y_train, X_test, y_test):
    tf.random.set_seed(32)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(4, tf.keras.activations.relu),
        tf.keras.layers.Dense(4, tf.keras.activations.relu),
        tf.keras.layers.Dense(10, tf.keras.activations.softmax),

    ])
    model.compile(tf.keras.optimizers.Adam(),
                  tf.keras.losses.SparseCategoricalCrossentropy(),
                  [tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    return model, history


def main():
    # PlotExample(11)
    # PlotExamplesRandomly()
    trainData_norm = trainData / 255.0
    testData_norm = testData / 255.0
    NeuralNetwork(trainData_norm, trainLabels, testData_norm, testLabels)

    pass


if __name__ == "__main__":
    main()
