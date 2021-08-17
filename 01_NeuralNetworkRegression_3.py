# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 01_NeuralNetworkRegression_3.py
@time: 2021-08-17 10:27
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    insuranceData = pd.read_csv("./resource/insurance.csv")

    # one hot encoding
    insuranceData_OneHot = pd.get_dummies(insuranceData)
    # print(insuranceData_OneHot.head())

    # create features and labels
    X = insuranceData_OneHot.drop("charges", axis=1)
    # print(X.head())
    y = insuranceData_OneHot["charges"]
    # print(y.head())

    # create training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    # build a neural network
    tf.random.set_seed(26)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(tf.keras.optimizers.Adam(),
                  'mae',
                  ['mae'])
    history = model.fit(X_train, y_train, epochs=2000, callbacks=[callback])
    model.evaluate(X_test, y_test)
    pd.DataFrame(history.history).plot()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()
    pass
