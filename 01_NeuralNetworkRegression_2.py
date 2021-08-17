# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 01_NeuralNetworkRegression_2.py
@time: 2021-08-16 11:31
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def PlotPredictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend()


if __name__ == "__main__":
    # create dataset
    X = tf.range(-100, 100, 4, dtype=tf.float32)
    y = X + 10

    # split dataset into training data and testing data
    splitLength = int(len(X) * 0.8)
    X_train = X[:splitLength]
    X_test = X[splitLength:]
    y_train = y[:splitLength]
    y_test = y[splitLength:]

    tf.random.set_seed(26)
    # create a model
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(2, 'relu'),
         tf.keras.layers.Dense(1)]
    )

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.mae,
                  metrics=['mse'])

    # fit the model
    model.fit(X_train, y_train, epochs=100, verbose=0)

    # save the model
    model.save('./savedModel')

    # load the model
    tf.keras.models.load_model('./savedModel')
    
    y_pred = tf.transpose(tf.constant(model.predict(X_test)))

    PlotPredictions(X_train, y_train, X_test, y_test, y_pred)

    mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred)
    print(mae)
    mse = tf.keras.metrics.mean_squared_error(y_test, y_pred)
    print(mse)
    plt.show()
    pass
