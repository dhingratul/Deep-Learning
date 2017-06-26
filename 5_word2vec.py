#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:17:51 2017

@author: dhingratul
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
# %matplotlib inline
from __future__ import print_function
import collections
import math
import numpy as np
# import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
# from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

"""
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
  # Download a file if not present, and make sure it's the right size.
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('/home/dhingratul/Documents/Dataset/text8.zip',
31344016)
"""
# Read the Data into a string


def read_data(filename):
    # Extract first file in the .zip as list of words
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
filename = '/home/dhingratul/Documents/Dataset/text8.zip'
words = read_data(filename)

vocab_size = 50000
# Build dictionary, replace rare words with UNK token


def build_dataset(words):
    count = [['UNK', -1]]
    # most_common(n), gives n most common words
    vocab = collections.Counter(words).most_common(vocab_size - 1)
    count.extend(vocab)
    dic = dict()
    for word, _ in count:
        dic[word] = len(dic)
    data = list()
    unk_ctr = 0
    for w in words:
        if w in dic:
            index = dic[w]
        else:
            index = 0
            unk_ctr = unk_ctr + 1  # UNK counter if it doesn't exist in dict
        data.append(index)
    count[0][1] = unk_ctr
    rev_dic = dict(zip(dic.values(), dic.keys()))
    return data, count, dic, rev_dic

data, count, dictionary, reverse_dictionary = build_dataset(words)

d_index = 0
# Generate training batch for skip-gram model


def generate_batch(batch_size, num_skips, skip_window):
    global d_index  # To access the copy of the global variable created
    # Assert: Test the condition, and trigger an error if the it is false
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # span === [skip_window target skip_window]
    span = 2 * skip_window + 1  # +1 for target
    # Initialize a double-ended queue with O(1) ops
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[d_index])
        d_index = (d_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[d_index])
        d_index = (d_index + 1) % len(data)
    return batch, labels

# Train a skip-gram model
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
"""
We pick a random validation set to sample nearest neighbors. here we limit the
validation samples to the words that have a low numeric ID, which by
construction are also the most frequent.
"""
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.
graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
    # Input Data
    train_data = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples)
    # Variables
    embeddings = tf.Variable(tf.random_uniform(
            [vocab_size, embedding_size], -1.0, 1.0))  # shape, minval, maxval
    softmax_w = tf.Variable(tf.truncated_normal(
            [vocab_size, embedding_size],
            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_b = tf.Variable(tf.zeros([vocab_size]))
    # Model - Look up for embeddings for input
    embed = tf.nn.embedding_lookup(embeddings, train_data)
    # Softmax loss using sample of negative labels each time
    # S.Softmax is a faster way to train softmax over huge number of classes
    loss_intermed = tf.nn.sampled_softmax_loss(
            weights=softmax_w, biases=softmax_b, inputs=embed,
            labels=train_labels, num_samples=num_sampled,
            num_classes=vocab_size)
    loss = tf.reduce_mean(loss_intermed)
    # Optimizer
    """ Optimizes both softmax_weights and embeddings, as embeddings are
    defined as a variable, and minimize method modifies all varibales
    """
    lr = 1.0
    optimizer = tf.train.AdagradOptimizer(lr).minimize(loss)
    # Similarity b/w mini-batches and all embeddings using Cosine distance
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    norm_embeddings = embeddings/norm
    valid_embeddings = tf.nn.embedding_lookup(norm_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(norm_embeddings))

# Access to graph
num_steps = 100001
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print("TF Graph Initialized")
    average_loss = 0
    for i in range(num_steps):
        batch_data, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
        feed_dict = {train_data: batch_data, train_labels: batch_labels}
        _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if i % 2000 == 0:
            if i > 0:
                average_loss = average_loss / 2000
                print('Average loss at step %d: %f' % (i, average_loss))
        if i % 10000 == 0:
            sim = similarity.eval()
            # Random set of words to evaluate similarit on (16)
            for j in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[j]]
                top_k = 8  # Number of NN
                NN = (-sim[i, :]).argsort()[1: top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[NN[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = norm_embeddings.eval()
