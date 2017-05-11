"""Tutorial on how to create a denoising autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import math
from utils import corrupt


# %%
def autoencoder(dimensions=[784, 512, 256, 128, 64, 32]):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')

    # Probability that we will corrupt input.
    # This is the essence of the denoising autoencoder, and is pretty
    # basic.  We'll feed forward a noisy input, allowing our network
    # to generalize better, possibly, to occlusions of what we're
    # really interested in.  But to measure accuracy, we'll still
    # enforce a training signal which measures the original image's
    # reconstruction cost.
    #
    # We'll change this to 1 during training
    # but when we're ready for testing/production ready environments,
    # we'll put it back to 0.
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    # Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    # latent representation
    z = current_input
    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
    return {'x': x, 'z': z, 'y': y,
            'corrupt_prob': corrupt_prob,
            'cost': cost, 
            'encoder': encoder}

# %%

def pca(ae, dimensions):
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    # current_input = corrupt(x) * ae['corrupt_prob'] + x * (1 - ae['corrupt_prob'])
    current_input = x

    # Build the encoder
    encoder = ae['encoder']
    encoder.reverse()
    for layer_i, n_output in enumerate(dimensions[1:]):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # latent representation
    z = current_input
    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    # now have the reconstruction through the network
    y = current_input
    return {'x': x, 'y': y}

def test_mnist():
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    dim = [784, 512, 256, 128, 64, 32]
    ae = autoencoder(dimensions=dim)

    # %%
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    batch_size = 100
    n_epochs = 5
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={
                ae['x']: train, ae['corrupt_prob']: [1.0]})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train, ae['corrupt_prob']: [1.0]}))

    # %%
    # Plot example reconstructions
    n_examples = 15
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm, ae['corrupt_prob']: [0.0]})
    p = pca(ae, dim)
    sess.run(tf.initialize_all_variables())
    p_recon = sess.run(p['y'], feed_dict={p['x']: test_xs_norm})
    fig, axs = plt.subplots(4, n_examples, figsize=(10, 2))

    for example_i in range(n_examples):
        diff = test_xs[example_i, :] - [recon[example_i, :] + mean_img]
        axs[0][example_i].imshow(np.reshape(test_xs[example_i, :], (28, 28)), cmap='gray', interpolation='nearest')
        axs[0][example_i].axis('off')
        axs[1][example_i].imshow(np.reshape([recon[example_i, :] + mean_img], (28, 28)), cmap='gray', interpolation='nearest')
        axs[1][example_i].axis('off')
        axs[2][example_i].imshow(np.reshape(diff, (28, 28)), cmap='gray', interpolation='nearest')
        axs[2][example_i].axis('off')
        axs[3][example_i].imshow(np.reshape([p_recon[example_i, :] + mean_img], (28, 28)), cmap='gray', interpolation='nearest')
        axs[3][example_i].axis('off')

    fig.show()
    plt.draw()
    plt.waitforbuttonpress()
    plt.savefig('test.png', bbox_inches='tight')

if __name__ == '__main__':
    test_mnist()