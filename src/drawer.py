# ========================================
# [] File Name : drawer.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Training a VAE to draw the MNIST characters dataset.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist as input_data
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# Load the dataset
MNIST = read_data_sets('MNIST_data')

# Reset graph sessions on the RAM
tf.reset_default_graph()

BATCH_SIZE = 64

# Batches of MNIST Characters
X_INPUT = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
Y_OUTPUT = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
Y_FLATTENED = tf.reshape(Y_OUTPUT, shape=[-1, 28*28])
KEEP_PROB = tf.placeholder(dtype=tf.float32, shape=(), name='KEEP_PROB')

# Essential constants
DEC_IN_CHANNELS = 1
NUM_LATENS = 8
RESHAPED_DIM = [-1, 7, 7, DEC_IN_CHANNELS]
INPUTS_DECODER = 49 * DEC_IN_CHANNELS / 2

def encoder(x_input, keep_prob):
    """
        Applies 3 layers of convolutions to the input data with dropout
    """
    with tf.variable_scope("encoder"):

        # Flatten the input
        x = tf.reshape(x_input, shape=[-1, 28, 28, 1])

        # CONV L1
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.leaky_relu)
        # L1 DROPOUT
        x = tf.nn.dropout(x, keep_prob)

        # CONV L2
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.leaky_relu)
        # L2 DROPOUT
        x = tf.nn.dropout(x, keep_prob)

        # CONV L3
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.leaky_relu)
        # L3 DROPOUT
        x = tf.nn.dropout(x, keep_prob)
        
        x = tf.contrib.layers.flatten(x)
        x_dense = tf.layers.dense(x, units=NUM_LATENS)
        sd = 0.5 * x_dense
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], NUM_LATENS]))
        z = x_dense + tf.multiply(epsilon, tf.exp(sd))

        return z, x_dense, sd

