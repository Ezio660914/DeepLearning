# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 07_TimeSeries_1.py
@time: 2021-08-31 13:48
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

tf.get_logger().setLevel('ERROR')

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")

dataDir = "resource/BTC_USD_2014-11-04_2021-08-31-CoinDesk.csv"


def main():
    # import time series with pandas
    df = pd.read_csv(dataDir,
                     parse_dates=["Date"],
                     index_col=["Date"])
    priceDf = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    print(priceDf.head())
    priceDf.plot(figsize=(10, 7), color="orange")
    plt.ylabel("BTC Price")
    plt.title("Price of Bitcoin")
    plt.legend(fontsize=14)
    plt.show()
    pass


if __name__ == "__main__":
    main()
