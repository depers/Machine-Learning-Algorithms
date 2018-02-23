# -*- coding: utf-8 -*-
__author__ = 'fengxiao'
__date__ = '2018/2/23 11:43'


import numpy as np
from math import sqrt
from collections import Counter


def kNN_classify(k, X_train, y_train, x):

    assert 1 <= k <= X_train.shape[0], 'k must be valid'
    assert X_train.shape[0] == y_train.shape[0], \
        'the size of X_train must equal to the size of y_train'
    assert X_train.shape[1] == x.shape[0], \
        'the feature number of x must be equal to X_train'

    distances = [sqrt(np.sum((x_tain - x)**2)) for x_tain in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[neighbor] for neighbor in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]