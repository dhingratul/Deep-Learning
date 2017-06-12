#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:58:16 2017

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


def reformat(data, labels):
    """
    Converts the data into a flat matrix, and labels into one-hot encoding
    """
    data = data.reshape((-1, image_size * image_size)).astype(np.float32)
    # -1:size being inferred from the parameters being passed
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return data, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# Training with tf
batch_size = 128
graph = tf.Graph()
with graph.as_default():
    # Use placeholder instead, that is fed at run time
    tf_train_data = tf.placeholder(tf.float32,
                                   shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_labels))
    tf_valid_data = tf.constant(valid_dataset)
    tf_test_data = tf.constant(test_dataset)
    # Variables are the parameters that are trained: Weights and Biases
    # Initialize weights to random values, using truncated normal distribution
    weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    # Training computation
    logits = tf.matmul(tf_train_data, weights) + biases
    # Softmax loss
    loss_intermediate = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf_train_labels, logits=logits)
    # Take mean over the loss
    avg_loss = tf.reduce_mean(loss_intermediate)
    # Gradient Descent Optimizer
    lr = 0.5  # Learning rate
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(avg_loss)
    # Predictions
    train_pred = tf.nn.softmax(logits)
    valid_logits = tf.matmul(tf_valid_data, weights) + biases
    valid_pred = tf.nn.softmax(valid_logits)
    test_logits = tf.matmul(tf_test_data, weights) + biases
    test_pred = tf.nn.softmax(test_logits)


def accuracy(predictions, labels):
    """ Outputs the accuracy based on gnd truth and predicted labels"""
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            labels.shape[0])
# Initialize the graph defined above
step_size = 3001
with tf.Session(graph=graph) as session:
    # Initialize weights
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(step_size):
        # Pick a randomized offset within training data
        offset = ((step * batch_size) % (train_labels.shape[0] - batch_size))
        # Generate a mini-batch`
        mb_data = train_dataset[offset:(offset + batch_size), :]
        mb_labels = train_labels[offset:(offset + batch_size), :]
        # Create a Dictionary to feed to mini batch
        feed_dict = {tf_train_data: mb_data, tf_train_labels: mb_labels}
        _, l, pred = session.run([optimizer, avg_loss, train_pred],
                                 feed_dict=feed_dict)
        if step % 500 == 0:
            print("MB Loss at step %d: %f" % (step, l))
            print("MB Accuracy: %0.1f%%"
                  % accuracy(pred, mb_labels))
            # Calling .eval() on valid_prediction is basically like calling
            # run(), but just to get that one numpy array Note that it
            # recomputes all its graph dependencies.
            print('Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(),
                                                           valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(),
                                                     test_labels))
