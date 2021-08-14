# Import TensorFlow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf

print(tf.__version__)
# Create tensors with tf.constant()
scalar = tf.constant(21)
print(scalar)
print(scalar.ndim)

vector = tf.constant([21, 21])
print(vector)
print(vector.ndim)

matrix = tf.constant([[10, 7],
                      [7, 10]])
matrix2 = tf.constant([[10., 23.],
                       [3., 22.],
                       [9., 15.]], dtype=tf.float16)

tensor = tf.constant([[[1, 2, 3], [4, 5, 6]],
                      [[7, 8, 9], [10, 11, 12]],
                      [[13, 14, 15], [16, 17, 18]]])
print(tensor)

# Create tensors with tf.Variable
ts1 = tf.Variable([10, 654])
print(ts1)
ts1[0].assign(13)
print(ts1)

# Create random tensors
r1 = tf.random.Generator.from_seed(2)
ts2 = r1.uniform(shape=(3, 2))
print(ts2)
ts3 = r1.normal(shape=(2, 3))
print(ts3)

# Shuffle the order of elements in a tensor
ts4 = tf.constant([[10, 7],
                   [7, 10]])
tf.random.shuffle(ts4)
print(ts4)

tf.random.set_seed(32)  # global level random seed
tf.random.shuffle(ts4, seed=32)  # operation level random seed

# other ways to create tensor
print(tf.ones([2, 4]))
print(tf.zeros([3, 2]))
# Turn NumPy arrays into tensors
import numpy as np

n1 = np.arange(1, 26, dtype=np.int32)
ts5 = tf.constant(n1)
ts6 = tf.constant(n1, shape=(5, 5))
print(ts5, ts6)



