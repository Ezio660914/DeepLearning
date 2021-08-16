# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 01_NeuralNetworkRegression_1.py
@time: 2021-08-15 10:48
"""

"""
regression with neural network
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # create data to view and fit
    X = np.array([-7, -4, -1, 2, 5, 8, 11, 14], dtype=np.float32)
    y = np.array([3, 6, 9, 12, 15, 18, 21, 24], dtype=np.float32)
    plt.scatter(X, y)
    # plt.show()
    # create a demo X and y prediction problem
    # set random seed
    tf.random.set_seed(15)

    # create a model using the sequential api
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    # compile the model
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=['mae'])

    # fit the model
    model.fit(X, y, epochs=5)

    # try and make prediction using our model
    xp = model.predict([17.0])
    print(xp)

    # improve the model
    # rebuild our model with a larger one
    # create a model using the sequential api
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # compile the model
    model.compile(loss='mae',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mae'])

    # fit the model
    model.fit(X, y, epochs=100)

    # try and make prediction using our model
    xp = model.predict([i for i in range(17, 27)])
    print(xp)

    pass
