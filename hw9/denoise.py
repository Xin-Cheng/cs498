"""Tutorial on how to create a denoising autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import math
from utils import corrupt
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as plt

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

def test_mnist():
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    
    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    dim = [784, 512, 256, 128, 64, 32]
    n_examples = 10000
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
    n_epochs = 20
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={
                ae['x']: train, ae['corrupt_prob']: [1.0]})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train, ae['corrupt_prob']: [1.0]}))
    
    # %%
    # Plot example reconstructions
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm, ae['corrupt_prob']: [0.0]})

    # fig, axs = plt.subplots(3, n_examples, figsize=(10, 2))
    diff = np.zeros((28*28))
    diff_arr = []
    for example_i in range(n_examples):
        curr_diff = test_xs[example_i, :] - [recon[example_i, :] + mean_img]
        diff = np.add(diff , curr_diff)
        diff_arr.append(curr_diff[0])

        # Xhat = np.dot(pca.transform(curr_diff), pca.components_[nComp,:])
        # Xhat += mu
    mean_diff = diff/n_examples
    zero_mean = np.transpose(np.array(diff_arr) - mean_diff)
    different_scale(zero_mean, mean_diff)


def different_scale(zero_mean, mean_diff):
    fig = plt.figure()
    plt.subplot(1, 6, 1)
    plt.imshow(np.reshape(mean_diff, (28, 28)), cmap='gray', interpolation='nearest')
    plt.title('Mean Residue')
    plt.axis('off')

    u,s,v = LA.svd(zero_mean)
    for i in range(5):
        dataReduced = np.dot(np.transpose(u)[i], np.reshape(mean_diff, (784, 1)))
        zeros = np.zeros((784, 1))
        zeros[i][0] = dataReduced
        dataReconstruct = np.dot(u,zeros)
        plt.subplot(1, 6, i + 2)
        plt.imshow(np.reshape(dataReconstruct, (28, 28)), cmap='gray', interpolation='nearest')
        name = 'PC' + str(i + 1)
        plt.title(name)
        plt.axis('off')

    fig.show()
    plt.draw()
    plt.savefig('individual_scale.png', bbox_inches='tight')

if __name__ == '__main__':
    test_mnist()