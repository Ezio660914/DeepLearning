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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


def PlotConfusionMatrix(y_test, y_preds):
    figSize = (10, 10)

    # Create the confusion matrix
    cm: np.ndarray = confusion_matrix(y_test, tf.round(y_preds))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=figSize)
    # Create a matrix plot
    cax = ax.matshow(cm,
                     cmap=plt.cm.get_cmap(
                         'Blues'))  # https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.matshow.html
    fig.colorbar(cax)

    # Create classes
    classes = False

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(20)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=15)
    plt.show()


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
    # plt.show()


def main():
    samplesAmount = 1000
    X, y = make_circles(samplesAmount, noise=0.03, random_state=13)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)
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
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.01),
                  tf.keras.losses.BinaryCrossentropy(),
                  [tf.keras.metrics.BinaryAccuracy()])

    # introduce a learning rate callback
    # lrScheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 20))

    # history = model.fit(X_train, y_train, epochs=100, callbacks=[lrScheduler])
    history = model.fit(X_train, y_train, epochs=20)
    model.evaluate(X_test, y_test)

    # make prediction
    y_pred = model.predict(X_test)
    y_pred = tf.round(y_pred)

    # plot confusion metrix
    PlotConfusionMatrix(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Train')
    PlotDecisionBoundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title('Test')
    PlotDecisionBoundary(model, X_test, y_test)
    pd.DataFrame(history.history).plot()
    plt.show()
    pass


if __name__ == "__main__":
    # main
    main()
