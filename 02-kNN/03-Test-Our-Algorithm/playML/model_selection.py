# -*- coding: utf-8 -*-
__author__ = 'fengxiao'
__date__ = '2018/2/23 12:49'

import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """
    将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test
    :param X: 训练数据集
    :param y: 训练数据集的“标记”或“答案”
    :param test_ratio: 测试数据集所占的比例
    :param seed: 随机种子
    :return:
    """

    assert X.shape[0] == y.shape[0], \
        'the size of X must be equal to the size of y'
    assert 0.0 <= test_ratio <= 1.0, \
        'test_ratio must be valid'

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
