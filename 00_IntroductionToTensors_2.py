# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 00_IntroductionToTensors_2.py
@time: 2021-08-14 17:14
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf

if __name__ == "__main__":
    # get information from tensors
    matrix1 = tf.constant([[10., 23.],
                           [3., 22.],
                           [9., 15.]], dtype=tf.float16)
    print(matrix1)
    # Shape, Rank, Axis or dimension, Size
    r1 = tf.random.Generator.from_seed(1)
    rank_4_tensor = r1.uniform(shape=(2, 3, 4, 5))
    print(rank_4_tensor)
    print(rank_4_tensor[0])
    print(rank_4_tensor[-1])
    print(rank_4_tensor.shape)
    print(rank_4_tensor.ndim)
    print(tf.size(rank_4_tensor).numpy())
    print(tf.size(rank_4_tensor))

    # indexing tensors
    print(rank_4_tensor[:2, :2, :2, :2])  # get the first 2 elements of each dimension
    print(rank_4_tensor[:1, :1, :1])

    rank_2_tensor = tf.constant([[1, 2], [3, 4]])
    print(rank_2_tensor[:, -1])
    # add in extra dimension to rank 2 tensor
    rank_3_tensor1 = rank_2_tensor[..., tf.newaxis]
    print(rank_3_tensor1)
    rank_3_tensor2 = rank_2_tensor[:, :, tf.newaxis]
    print(rank_3_tensor2)
    rank_3_tensor3 = tf.expand_dims(rank_2_tensor, axis=-1)
    print(rank_3_tensor3)

    # manipulating tensors
    ts = tf.constant([[1, 2, 5],
                      [7, 2, 1],
                      [3, 3, 3]])
    print(ts)
    print(ts + 10)
    print(ts * 10)
    print(ts - 10)
    print(ts / 2)
    tf.multiply(ts, 10)
    tf.add(ts, 9)
    tf.math.subtract(ts, 100)
    tf.math.divide(ts, 3)

    # matrix multiplication
    ts2 = tf.constant([[3, 5],
                       [6, 7],
                       [1, 8]])
    print(tf.matmul(ts, ts2))
    print(ts @ ts2)
    print(tf.tensordot(ts, ts2, 1))

    # change the datatype of a tensor
    ts3 = tf.constant([1.3, 2.8])
    print(ts3)
    ts3 = tf.cast(ts3, tf.float16)
    print(ts3)
    ts3 = tf.cast(ts3, tf.int32)
    print(ts3)
