# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 07_TimeSeries_2.py
@time: 2021-08-31 20:10
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
epochs = 100
batchSize = 128


def MeanAbsoluteScaledError(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    return mae / mae_naive_no_season


def EvaluatePreds(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae = keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.math.sqrt(mse)
    mape = keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = MeanAbsoluteScaledError(y_true, y_pred)
    return dict(
        mae=mae.numpy(),
        mse=mse.numpy(),
        rmse=rmse.numpy(),
        mape=mape.numpy(),
        mase=mase.numpy()
    )


def GetLabelledWindow(x, horizon):
    """create labels for windowed dataset
    E.g. if horizon=1
    input: [0,1,2,3,4,5,6,7]
    output: ([0,1,2,3,4,5,6], [7])"""
    return x[:, :-horizon], x[:, -horizon:]


def MakeWindows(x, windowSize, horizon):
    """Turn 1D array into a 2D array of sequential labelled windows of
    windowSize with horizon size labels"""
    # create a window of specific window size (add the horizon on the end for labelling later)
    windowStep = np.expand_dims(np.arange(windowSize + horizon), axis=0)
    # create a 2D array of multiple window steps
    windowIndexes = windowStep + np.expand_dims(np.arange(len(x) - (windowSize + horizon) + 1), axis=0).transpose()
    # index on the target array (a time series) with 2D array of multiple window steps
    windowedArray = x[windowIndexes]
    # get the labelled windows
    windows, labels = GetLabelledWindow(windowedArray, horizon)
    return windows, labels


def MakeTrainTestSplit(windows, labels, testSplit=0.2):
    """split matching paris of window and labels into train and test split"""
    splitSize = int(len(windows) * (1. - testSplit))
    trainWindows = windows[:splitSize]
    trainLabels = labels[:splitSize]
    testWindows = windows[splitSize:]
    testLabels = labels[splitSize:]
    return trainWindows, testWindows, trainLabels, testLabels


def BuildDenseModel():
    inputs = keras.Input(shape=(windowSize,))
    net = keras.layers.Dense(512)(inputs)
    net = keras.layers.Activation(keras.activations.relu, dtype=tf.float32)(net)
    net = keras.layers.Dense(128)(net)
    net = keras.layers.Activation(keras.activations.relu, dtype=tf.float32)(net)
    outputs = keras.layers.Dense(horizon)(net)
    model = keras.Model(inputs, outputs)
    return model


def BuildCNNModel():
    inputs = keras.Input(shape=(windowSize,))
    net = keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(inputs)
    net = keras.layers.Conv1D(filters=128,
                              kernel_size=3,
                              padding="causal")(net)
    net = keras.layers.Activation(keras.activations.relu, dtype=tf.float32)(net)
    net = keras.layers.GlobalAveragePooling1D()(net)
    net = keras.layers.Dense(128)(net)
    net = keras.layers.Activation(keras.activations.relu, dtype=tf.float32)(net)
    net = keras.layers.Dense(16)(net)
    net = keras.layers.Activation(keras.activations.relu, dtype=tf.float32)(net)
    outputs = keras.layers.Dense(horizon)(net)
    model = keras.Model(inputs, outputs)
    return model


def BuildLSTMModel():
    inputs = keras.Input(shape=(windowSize))
    net = keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(inputs)
    net = keras.layers.LSTM(128, "relu", return_sequences=True)(net)
    net = keras.layers.LSTM(128, "relu")(net)
    net = keras.layers.Dense(128, "relu")(net)
    net = keras.layers.Dense(16, "relu")(net)
    outputs = keras.layers.Dense(horizon)(net)
    model = keras.Model(inputs, outputs)
    return model


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


class BlockReward:
    # add in block reward feature to our data
    blockReward = [25., 12.5, 6.25]
    blockRewardDate = [np.datetime64("2012-11-28"),
                       np.datetime64("2016-07-09"),
                       np.datetime64("2020-05-18")]

    @staticmethod
    def AddFeature(priceDf):
        blockRewardDays = [max(0, (date - priceDf.index[0]).days) for date in BlockReward.blockRewardDate] + [
            len(priceDf)]
        print(blockRewardDays)
        priceDf["BlockReward"] = None
        for i in range(3):
            priceDf.iloc[blockRewardDays[i]:blockRewardDays[i + 1], 1] = BlockReward.blockReward[i]
        print(priceDf)
        print(priceDf["BlockReward"].value_counts())
        # normalize the data using min-max scale
        priceDf = pd.DataFrame(minmax_scale(priceDf[["Price", "BlockReward"]]), index=priceDf.index,
                               columns=priceDf.columns)
        return priceDf


def main():
    # import time series with pandas
    df = pd.read_csv(dataDir,
                     parse_dates=["Date"],
                     index_col=["Date"])
    # add in block reward feature to data frame
    priceDf = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    priceDf = BlockReward.AddFeature(priceDf)
    for i in range(len(priceDf)):
        priceDf.shift
    # priceDf.plot(figsize=(10, 7), color="orange")
    # plt.ylabel("BTC Price")
    # plt.title("Price of Bitcoin")
    # plt.legend(fontsize=14)
    # plt.show()

    # define window and horizon

    priceData = priceDf["Price"].to_numpy()
    windows, labels = MakeWindows(priceData, windowSize, horizon)
    for i in range(3):
        print((windows[i], labels[i]))

    # turning windows into training and test sets
    trainWindows, testWindows, trainLabels, testLabels = MakeTrainTestSplit(windows, labels, 0.2)
    # create checkpoint callback
    ckptCallback = keras.callbacks.ModelCheckpoint(checkpointDir,
                                                   save_best_only=True,
                                                   save_weights_only=True)
    # create LSTM model, window=7, horizon=1
    model = BuildLSTMModel()
    model.compile(keras.optimizers.Adam(),
                  keras.losses.mae,
                  metrics=["mse"])
    model.summary()

    history = model.fit(trainWindows, trainLabels,
                        epochs=epochs,
                        batch_size=batchSize,
                        callbacks=ckptCallback,
                        validation_data=(testWindows, testLabels))
    lossDict = dict(loss=history.history["loss"],
                    val_loss=history.history["val_loss"])
    mseDict = dict(mse=history.history["mse"],
                   val_mse=history.history["val_mse"])
    plt.subplot(1, 2, 1)
    plt.plot(pd.DataFrame(lossDict), label=lossDict.keys())
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(pd.DataFrame(mseDict), label=mseDict.keys())
    plt.legend()
    # make predictions with the best saved model
    plt.figure()
    model.load_weights(checkpointDir)
    model.evaluate(testWindows, testLabels)
    testPred = tf.squeeze(model.predict(testWindows))
    plot_time_series(priceDf.index[-len(testWindows):].to_numpy(), testPred, label="pred")
    plot_time_series(priceDf.index[-len(testWindows):].to_numpy(), testLabels, '-', label="true")
    plt.show()
    pass


if __name__ == "__main__":
    main()
