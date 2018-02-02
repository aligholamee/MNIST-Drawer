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

