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


def reformat(data, labels, num_labels, image_size):
    """
    Converts the data into a flat matrix, and labels into one-hot encoding
    """
    data = data.reshape((-1, image_size * image_size)).astype(np.float32)
    # -1:size being inferred from the parameters being passed
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return data, labels

image_size = 28
num_labels = 10
train_dataset, train_labels = reformat(train_dataset, train_labels,
                                       image_size, num_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels,
                                       image_size, num_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels,
                                     image_size, num_labels)

# Training with tf
batch_size = 5000
graph = tf.Graph()
with graph.as_default():
    # Data is treated as tf.constant in the tensorflow graph
    tf_train_data = train_dataset[:batch_size, :]
    tf_train_data = tf.constant(tf_train_data)
    tf_train_labels = tf.constant(train_labels[:batch_size])
    tf_valid_data = tf.constant(valid_dataset)
    tf_test_data = tf.constant(test_dataset)
    # Variables are the parameters that are trained: Weights and Biases
    # Initialize weights to random values, using truncated normal distribution
    weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    # Training computation
    logits = tf.matmul(tf_train_data * weights) + biases
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

step_size = 500


def accuracy(predictions, labels):
    """ Outputs the accuracy based on gnd truth and predicted labels"""
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            labels.shape[0])
# Initialize the graph defined above
with tf.Session(graph=graph) as session:
    # Initialize weights
    tf.global_variables_initializer().run()
    for step in range(step_size):
        # Run the computations. We tell .run() that we want to run the
        # optimizer,and get the loss value and the training predictions
        # returned as numpy arrays.
        _, l, pred = session.run([optimizer, avg_loss, train_predictions])
        if step % 100 == 0:
            print("Loss at step %d: %f" % (step, l))
            print("Training Accuracy: %0.1f%%"
                  % accuracy(pred, train_labels[:batch_size, :]))
            # Calling .eval() on valid_prediction is basically like calling
            # run(), but just to get that one numpy array Note that it
            # recomputes all its graph dependencies.
            print('Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(),
                                                           valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(),
                                                     test_labels))
