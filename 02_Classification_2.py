# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 02_Classification_1.py
@time: 2021-08-17 18:00
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles


def PlotDecisionBoundary(model, X, y):
    """
    plots the decision boundary created by a model predicting on X.
    :param model:
    :param X:
    :param y:
    :return:
    """
    # define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    # create x value
    x_in = np.c_[xv.ravel(), yv.ravel()]  # stack 2D arrays together
    # make predictions
    y_pred = model.predict(x_in)
    # check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        # reshape
        y_pred = np.argmax(y_pred, axis=1).reshape(xv.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xv.shape)
    # plot the decision boundary
    plt.contourf(xv, yv, y_pred, cmap=plt.cm.get_cmap('RdYlBu'), alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.get_cmap('RdYlBu'))
    plt.xlim(xv.min(), xv.max())
    plt.ylim(yv.min(), yv.max())
    plt.show()


def main():
    samplesAmount = 1000
    X, y = make_circles(samplesAmount, noise=0.03, random_state=13)
    circles = pd.DataFrame({'X0': X[:, 0],
                            'X1': X[:, 1],
                            'label:': y})
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    tf.random.set_seed(12)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, tf.keras.activations.relu),
        tf.keras.layers.Dense(10, tf.keras.activations.relu),
        tf.keras.layers.Dense(1, tf.keras.activations.sigmoid),
    ])
    model.compile(tf.keras.optimizers.Adam(),
                  tf.keras.losses.BinaryCrossentropy(),
                  [tf.keras.metrics.BinaryAccuracy()])
    model.fit(X, y, epochs=100)
    PlotDecisionBoundary(model, X, y)
    pass


if __name__ == "__main__":
    # main
    main()
