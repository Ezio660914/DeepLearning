# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 01_NeuralNetworkRegression_4.py
@time: 2021-08-17 11:56
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

if __name__ == "__main__":
    # main
    insuranceData = pd.read_csv("./resource/insurance.csv")

    # use transformer to normalise dataset
    ct = make_column_transformer(
        (MinMaxScaler(), ['age', 'bmi', 'children']),
        (OneHotEncoder(handle_unknown='ignore'), ['sex', 'smoker', 'region'])
    )

    # create x and y
    X = insuranceData.drop('charges', 1)
    y = insuranceData['charges']

    # build train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

    # fit the column transformer to our training data
    ct.fit(X)

    # transform training and test data with normalization and one hot encoder
    X_train_normal = ct.transform(X_train)
    X_test_normal = ct.transform(X_test)
    # build neural network
    callback = tf.keras.callbacks.EarlyStopping('loss', patience=3)
    tf.random.set_seed(23)
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
    history = model.fit(X_train_normal, y_train, epochs=2000, callbacks=[callback])
    model.evaluate(X_test_normal, y_test)
    pd.DataFrame(history.history).plot()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    pass
