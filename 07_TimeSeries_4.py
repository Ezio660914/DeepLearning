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


def main():
    exit(0)


if __name__ == "__main__":
    main()
