# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 06_NLP_1.py
@time: 2021-08-27 17:42
"""
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import pathlib
import datetime

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")


def main():
    pass


if __name__ == "__main__":
    main()
