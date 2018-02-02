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

# Load the dataset
import tensorflow.examples.tutorials.mnist as input_data
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

MNIST = read_data_sets('MNIST_data')