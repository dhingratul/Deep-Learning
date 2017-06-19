#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:02:35 2017

@author: dhingratul
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


def unPickle(pickle_file):
    """
    Unpickles the data file into tr, te, and validation data
    """
    with open(pickle_file, 'rb') as f:
        datasets = pickle.load(f)
        test_dataset = datasets['test_dataset']
        test_labels = datasets['test_labels']
        train_dataset = datasets['train_dataset']
        train_labels = datasets['train_labels']
        valid_dataset = datasets['valid_dataset']
        valid_labels = datasets['valid_labels']
    return test_dataset, test_labels, train_dataset, train_labels,\
        valid_dataset, valid_labels

pickle_file = "/home/dhingratul/Documents/Dataset/notMNIST.pickle"
test_dataset, test_labels, train_dataset, train_labels, valid_dataset,\
    valid_labels = unPickle(pickle_file)

"""
Reformat data as per the requirements of the program, data as a flat matrix,
and label as one hot encoded vector
"""

image_size = 28
num_labels = 10
num_channels = 1  # Grayscale


def reformat(data, labels):
    """
    Converts the data into a flat matrix, and labels into one-hot encoding
    """
    data = data.reshape(
            (-1, image_size, image_size, num_channels)).astype(np.float32)
    # -1:size being inferred from the parameters being passed
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return data, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


def accuracy(predictions, labels):
    """ Outputs the accuracy based on gnd truth and predicted labels"""
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            labels.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
# Network: 2conv->1FC
graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size,
                               image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    # Variables
    # Weights = ptch_size X ptch_size , input_depth, output_depth
    layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden],
            stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    def model(data):
        conv1 = tf.nn.conv2d(
                data, layer1_weights, [1, 2, 2, 1], padding='SAME'
                )
        """(input, filter, stride, padding)
        [1, stride, stride, 1] for NHWC fornat
        """
        hidden1 = tf.nn.relu(conv1 + layer1_biases)
        conv2 = tf.nn.conv2d(
                hidden1, layer2_weights, [1, 2, 2, 1], padding='SAME'
                )
        hidden2 = tf.nn.relu(conv2 + layer2_biases)
        # Reshaping for the FC layer
        shape_hd2 = hidden2.get_shape().as_list()
        # Flatten it out
        hidden2_rshp = tf.reshape(
                hidden2,
                [shape_hd2[0], shape_hd2[1] * shape_hd2[2] * shape_hd2[3]]
                )
        fc1 = tf.matmul(hidden2_rshp, layer3_weights)
        hidden3 = tf.nn.relu(fc1 + layer3_biases)
        predict = tf.matmul(hidden3, layer4_weights) + layer4_biases
        return predict
    # Training
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf_train_labels, logits=logits)
            )
    # Optimizer
    lr = 0.05
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    # Predictions
    train_pred = tf.nn.softmax(logits)
    valid_pred = tf.nn.softmax(model(tf_valid_dataset))
    test_pred = tf.nn.softmax(model(tf_test_dataset))

# Feeding data to the graph
num_steps = 1001
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print("Model Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels}
        _, l, pred = sess.run([optimizer, loss, train_pred],
                              feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(pred, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(),
                                                           valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_labels))
