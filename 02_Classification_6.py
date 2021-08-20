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


def PlotConfusionMatrix(y_test, y_preds, classes=None, figSize=(10, 10), textSize=15):
    from sklearn.metrics import confusion_matrix
    import itertools

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
    # set labels to be classes
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
    ax.xaxis.label.set_size(textSize)
    ax.yaxis.label.set_size(textSize)
    ax.title.set_size(textSize)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=textSize)
    plt.show()


def PlotRandomImage(model, images, trueLabels, classes):
    """picks a random image, plots it and labels it with a prediction and truth label"""
    import random
    # set up random integer
    i = random.randint(0, len(images))
    # create predictions and targets
    targetImage = images[i]
    predProbs = model.predict(targetImage.reshape(1, 28, 28))
    predLabel = classes[predProbs.argmax()]
    trueLabels = classes[trueLabels[i]]
    # plot the image
    plt.imshow(targetImage, cmap=plt.cm.get_cmap('binary'))
    # change the color of the titles depending on if the prediction is right or wrong
    if predLabel == trueLabels:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel(f'Pred:{predLabel} {100 * tf.reduce_max(predProbs):2.0f}% (True:{trueLabels})',
               color=color)
    plt.show()


def main():
    # PlotExample(11)
    # PlotExamplesRandomly()
    trainData_norm = trainData / 255.0
    testData_norm = testData / 255.0
    model, history = NeuralNetwork(trainData_norm, trainLabels, testData_norm, testLabels)
    y_probs = model.predict(testData_norm)
    y_preds = y_probs.argmax(axis=1)
    PlotConfusionMatrix(testLabels, y_preds, classNames, (15, 15), 10)
    PlotRandomImage(model, testData_norm, testLabels, classNames)
    pass


if __name__ == "__main__":
    main()
