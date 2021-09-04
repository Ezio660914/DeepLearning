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
intoFuture = 14
trainModel = False


def BuildModel(windowSize, horizon):
    inputs = keras.layers.Input(shape=(windowSize,))
    x = keras.layers.Dense(128, "relu")(inputs)
    x = keras.layers.Dense(128, "relu")(x)
    outputs = keras.layers.Dense(horizon)(x)
    model = keras.Model(inputs, outputs)
    return model


def MakeFutureForecasts(values, model: keras.Model, into_future, window_size):
    """make future forecasts into_future steps after values ends
    returns future forecasts as a list of float values"""
    futureForecast = []
    lastWindow = values[-window_size:]
    # make into_future number of predictions, altering the data which gets predicted on each
    for i in range(into_future):
        # predict on the last window then append it again
        futurePred = model.predict(tf.expand_dims(lastWindow, 0))
        print(f"Predicting on:\n{lastWindow} -> Prediction: {tf.squeeze(futurePred).numpy()}\n")
        futureForecast.append(tf.squeeze(futurePred).numpy())
        lastWindow = np.append(lastWindow, futurePred)[-window_size:]
    return futureForecast


def GetFutureDates(start_date, into_future, offset=1):
    """return array of datetime values ranging from start_date to start_date + into_future"""
    start_date = start_date + np.timedelta64(offset, "D")
    end_date = start_date + np.timedelta64(into_future, "D")
    return np.arange(start_date, end_date, dtype="datetime64[D]")


def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    """
    Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

    Parameters
    ---------
    timesteps : array of timesteps
    values : array of values across time
    format : style of plot, default "."
    start : where to start the plot (setting a value will index from start of timesteps & values)
    end : where to end the plot (setting a value will index from end of timesteps & values)
    label : label to show on plot of values
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)  # make label bigger
    plt.grid(True)


def main():
    # create N-Beats data inputs
    df = pd.read_csv(dataDir,
                     parse_dates=["Date"],
                     index_col=["Date"])
    # add in block reward feature to data frame
    priceDf = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    bitcoinPricesShifted = priceDf.copy()
    for i in range(windowSize):
        bitcoinPricesShifted[f"Price_{i + 1}"] = bitcoinPricesShifted["Price"].shift(i + 1)
    X = pd.DataFrame(bitcoinPricesShifted.dropna().drop("Price", axis=1), dtype=np.float32)
    y = pd.DataFrame(bitcoinPricesShifted.dropna()["Price"], dtype=np.float32)

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
    if trainModel:
        history = model.fit(allData,
                            epochs=epochs,
                            callbacks=[callback])
        pd.DataFrame(history.history["loss"]).plot()
        plt.show()
    model.load_weights(checkpointDir)
    model.evaluate(allData)
    futureForecast = MakeFutureForecasts(y, model, intoFuture, windowSize)
    futureForecast = np.insert(futureForecast, 0, y["Price"][-1])
    print(futureForecast)
    nextTimeSteps = GetFutureDates(X.index[-1], intoFuture)
    nextTimeSteps = np.insert(nextTimeSteps, 0, X.index[-1])
    print(nextTimeSteps)
    plt.figure(figsize=(10, 7))
    plot_time_series(priceDf.index, priceDf, start=2400, format="-", label="Acture")
    plot_time_series(nextTimeSteps, futureForecast, format="-", label="Predicted")
    plt.show()
    exit(0)


if __name__ == "__main__":
    main()
