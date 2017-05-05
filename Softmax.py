#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:49:09 2017

@author: dhingratul
Create a softmax function to translate Scores into probabilities
as a part of Lesson1, Quiz 10
"""
import numpy as np
import matplotlib.pyplot as plt
scores = [3.0, 1.0, 0.2]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x), axis=0)

print(softmax(scores))

# Plot softmax curves
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
