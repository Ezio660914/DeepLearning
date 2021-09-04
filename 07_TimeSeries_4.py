# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 07_TimeSeries_4.py
@time: 2021-09-03 19:54
"""
import os
import random
import matplotlib.colors
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import *
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pathlib
import datetime
from sklearn.preprocessing import minmax_scale

tf.get_logger().setLevel('ERROR')

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")

dataDir = "resource/BTC_USD_2014-11-04_2021-08-31-CoinDesk.csv"
checkpointDir = "./savedModel/TimeSeries/Conv1DModel"
horizon = 1
windowSize = 7
epochs = 5000
batchSize = 1024
nNeurons = 512
nLayers = 4
nStacks = 30


# create NBeatBlock custom layer
class NBeatsBlock(keras.layers.Layer):
    def __init__(self, *,
                 input_size: int,
                 theta_size: int,
                 horizon: int,
                 n_neurons: int,
                 n_layers: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # Block contains stack of 4 fully connected layers,
        # each has relu activation
        self.hidden = [keras.layers.Dense(n_neurons, "relu", name=f"FC_{i + 1}") for i in range(4)]
        # Output of block is a theta layer with linear activation
        self.theta_layer = keras.layers.Dense(theta_size, "linear", name="theta")

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        # output the backcast and the forecast from theta
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast


def BuildNBeatsModel(horizon,
                     windowSize,
                     nNeurons,
                     nLayers,
                     nStacks):
    inputSize = windowSize * horizon
    thetaSize = inputSize + horizon
    inputs = keras.layers.Input(shape=(inputSize,))
    initialBlockLayer = NBeatsBlock(input_size=inputSize,
                                    theta_size=thetaSize,
                                    horizon=horizon,
                                    n_neurons=nNeurons,
                                    n_layers=nLayers,
                                    name="InitialBlock")
    # create initial backcast and forecast input
    residuals, forecast = initialBlockLayer(inputs)
    # create stacks of block layers
    for i in range(nStacks - 1):
        backcast, blockForecast = NBeatsBlock(input_size=inputSize,
                                              theta_size=thetaSize,
                                              horizon=horizon,
                                              n_neurons=nNeurons,
                                              n_layers=nLayers,
                                              name=f"NBeatsBlock_{i}")(residuals)
        residuals = keras.layers.subtract([residuals, backcast], name=f"subtract_{i}")
        forecast = keras.layers.add([forecast, blockForecast], name=f"add_{i}")
    model = keras.Model(inputs, forecast)
    return model


def main():
    # create N-Beats data inputs
    df = pd.read_csv(dataDir,
                     parse_dates=["Date"],
                     index_col=["Date"])
    # add in block reward feature to data frame
    priceDf = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    bitcoinPricesNbeats = priceDf.copy()
    for i in range(windowSize):
        bitcoinPricesNbeats[f"Price_{i + 1}"] = bitcoinPricesNbeats["Price"].shift(i + 1)
    X = pd.DataFrame(bitcoinPricesNbeats.dropna().drop("Price", axis=1), dtype=np.float32)
    y = pd.DataFrame(bitcoinPricesNbeats.dropna()["Price"], dtype=np.float32)

    # make train and test sets
    splitSize = int(len(X) * 0.8)
    X_train, y_train = X[:splitSize], y[:splitSize]
    X_test, y_test = X[splitSize:], y[splitSize:]

    # time to make our dataset performant using tf.data api
    trainFeaturesDataset = tf.data.Dataset.from_tensor_slices(X_train)
    trainLabelsDataset = tf.data.Dataset.from_tensor_slices(y_train)
    testFeaturesDataset = tf.data.Dataset.from_tensor_slices(X_test)
    testLabelsDataset = tf.data.Dataset.from_tensor_slices(y_test)

    # combine labels and features by zipping together
    trainDataset = tf.data.Dataset.zip((trainFeaturesDataset, trainLabelsDataset))
    testDataset = tf.data.Dataset.zip((testFeaturesDataset, testLabelsDataset))

    # batch and prefetch
    trainDataset = trainDataset.batch(batch_size=batchSize).prefetch(tf.data.AUTOTUNE)
    testDataset = testDataset.batch(batch_size=batchSize).prefetch(tf.data.AUTOTUNE)

    # create model
    model = BuildNBeatsModel(horizon=horizon,
                             windowSize=windowSize,
                             nNeurons=nNeurons,
                             nLayers=nLayers,
                             nStacks=nStacks)
    model.compile(keras.optimizers.Adam(),
                  "mae",
                  ["mae"])
    model.summary()
    # create callback
    callback_1 = keras.callbacks.EarlyStopping(patience=200,
                                               restore_best_weights=True,
                                               verbose=1)
    callback_2 = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                   patience=100,
                                                   verbose=1)
    keras.utils.plot_model(model, show_shapes=True, dpi=300)
    # fit the model
    history = model.fit(trainDataset,
                        epochs=epochs,
                        validation_data=testDataset,
                        callbacks=[callback_1, callback_2],
                        verbose=0)
    model.evaluate(testDataset)
    exit(0)


if __name__ == "__main__":
    main()
