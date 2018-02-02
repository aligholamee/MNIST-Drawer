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

def decoder(sampled_z, keep_prob):
    """
        Regenerate an image using coded images
    """
    with tf.variable_scope("decoder"):
        x = tf.layers.dense(sampled_z, units=INPUTS_DECODER, activation=tf.nn.leaky_relu)
        x = tf.layers.dense(x, units=INPUTS_DECODER*2 + 1, activation=tf.nn.leaky_relu)

        x = tf.reshape(x, RESHAPED_DIM)

        # TRANSPOSED CONV 1-3 + DROPOUTS
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])

        return img

CODED_IMG, MN, SD = encoder(X_INPUT, KEEP_PROB)
DECODED_IMG = decoder(CODED_IMG, KEEP_PROB)

# Compute the image reconstruction loss
UN_RESHAPED = tf.reshape(DECODED_IMG, [-1, 28*28])
IMG_LOSS = tf.reduce_sum(tf.squared_difference(UN_RESHAPED, Y_FLATTENED), 1)
LATENT_LOSS = -0.5 * tf.reduce_sum(1.0 + 2.0*SD - tf.square(MN) - tf.expo(2.0*SD), 1)
LOSS = tf.reduce_mean(IMG_LOSS, LATENT_LOSS)
OPTIMIZER = tf.train.AdamOptimizer(0.0005).minimize(LOSS)

# Run the session
SESS = tf.Session()
SESS.run(tf.global_variables_initializer())

# Take the minibatches and feed the session dicitionary
for i in range(30000):
    BATCH = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=BATCH_SIZE)[0]]
    SESS.run(OPTIMIZER, feed_dict={X_INPUT: BATCH, Y_FLATTENED: BATCH, KEEP_PROB: 0.8})

    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigma = SESS.run([LOSS, DECODED_IMG, IMG_LOSS, LATENT_LOSS, MN, SD],
        feed_dict={X_INPUT: BATCH, Y_FLATTENED: BATCH, KEEP_PROB: 1.0})
        plt.imshow(np.reshape(BATCH[0], [28, 28]), cmap='gray')
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(d_ls))