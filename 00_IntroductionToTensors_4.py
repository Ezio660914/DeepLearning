# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 00_IntroductionToTensors_4.py
@time: 2021-08-14 21:34
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    # one-hot encoding
    itemList = [0, 1, 2, 3]
    ts1 = tf.one_hot(itemList, depth=len(itemList))
    print(ts1)

    # specity custom values for one hot encoding
    ts1 = tf.one_hot(itemList, depth=len(itemList), on_value=True, off_value=False)
    print(ts1)

    # more math operations
    ts2 = tf.range(1, 10)
    ts2 = tf.cast(ts2, tf.float16)
    print(ts2)
    print(tf.math.square(ts2))
    print(tf.math.sqrt(ts2))
    print(tf.math.log(ts2))

    # tensors and numpy
    ts3 = tf.constant(np.array([3., 4., 12.]))
    print(ts3)
    print(ts3.numpy())

    print(tf.config.list_physical_devices())

    pass
