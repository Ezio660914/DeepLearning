# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 00_IntroductionToTensors_3.py
@time: 2021-08-14 19:01
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    # aggregate tensors
    # get the absolute values
    ts1 = tf.constant(np.random.randint(0, 100, 5), dtype=tf.float16)
    print(tf.abs(ts1))
    # get the minimum
    print(tf.reduce_min(ts1))
    # get the maximum
    print(tf.reduce_max(ts1))
    # get the mean of a tensor
    print(tf.reduce_mean(ts1))
    # get the sum of a tensor
    print(tf.reduce_sum(ts1))
    # variance
    print(tf.math.reduce_variance(ts1))
    # standard deviation
    print(tf.math.reduce_std(ts1))
    # positional max and min
    ts2 = tf.random.uniform(shape=[2, 3])
    print(ts2)
    print(tf.argmax(ts2, axis=0))  # return the index
    print(tf.argmin(ts2, axis=0))

    # squeeze a tensor (remove all single dimension)
    ts3 = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
    print(ts3)
    ts3_squeezed = tf.squeeze(ts3)
    print(ts3_squeezed)
    pass
