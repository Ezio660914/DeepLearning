# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 08_TimeSeries_5.py
@time: 2021-09-04 19:12
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
checkpointDir = "./savedModel/TimeSeries/RealFutureModel"
horizon = 1
windowSize = 7
epochs = 5000
batchSize = 1024
nNeurons = 512
nLayers = 4
nStacks = 30


def BuildModel(windowSize, horizon):
    inputs = keras.layers.Input(shape=(windowSize,))
    x = keras.layers.Dense(128, "relu")(inputs)
    x = keras.layers.Dense(128, "relu")(x)
    outputs = keras.layers.Dense(horizon)(x)
    model = keras.Model(inputs, outputs)
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

    # turn data into datasets
    featuresDataset = tf.data.Dataset.from_tensor_slices(X)
    labelsDataset = tf.data.Dataset.from_tensor_slices(y)
    allData = tf.data.Dataset.zip((featuresDataset, labelsDataset))
    allData = allData.batch(batchSize).prefetch(tf.data.AUTOTUNE)

    # create model
    model = BuildModel(windowSize, horizon)
    model.compile(keras.optimizers.Adam(),
                  "mae",
                  ["mse"])
    callback = keras.callbacks.ModelCheckpoint(filepath=checkpointDir,
                                               monitor="loss",
                                               verbose=1,
                                               save_best_only=True,
                                               save_weights_only=True)
    history = model.fit(allData,
                        epochs=epochs,
                        callbacks=[callback])
    model.load_weights(checkpointDir)
    model.evaluate(allData)
    pd.DataFrame(history.history["loss"]).plot()
    plt.show()
    exit(0)


if __name__ == "__main__":
    main()
