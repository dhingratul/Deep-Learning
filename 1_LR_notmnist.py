#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:42:41 2017

@author: dhingratul
Performs Logistic Regression on notMNIST dataset from Udacity course on Deep
Learning
Input : Pickle file
Output: Performance of LR on given batch size
"""
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle


def unPickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        datasets = pickle.load(f)
    test_dataset = datasets['test_dataset']
    test_labels = datasets['test_labels']
    train_dataset = datasets['train_dataset']
    train_labels = datasets['train_labels']
    valid_dataset = datasets['valid_dataset']
    valid_labels = datasets['valid_labels']
    return test_dataset, test_labels, train_dataset, train_labels,
    valid_dataset, valid_labels

pickle_file = "/home/dhingratul/Documents/Dataset/notMNIST.pickle"
test_dataset, test_labels, train_dataset, train_labels = unPickle(pickle_file)

# Logistic Regression Model
batch_size = 10000
X_train = train_dataset[:batch_size].reshape(batch_size, 784)
Y_train = train_labels[:batch_size]
X_test = test_dataset.reshape(test_dataset.shape[0], 784)
Y_test = test_labels
model = LogisticRegression()
model = model.fit(X_train, Y_train)
# Testing Accuracy
print(model.score(X_test, Y_test))
