# -*- coding: utf-8 -*-
__author__ = 'fengxiao'
__date__ = '2018/2/23 12:49'

import numpy as np
from math import sqrt
from collections import Counter


class KNNClassify:

    def __init__(self, k):
        """
        初始化KNN分类器
        """
        assert k >= 1, 'k must be valid'
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """
        根据训练数据集X_train和y_train训练kNN分类器 
        """
        assert X_train.shape[0] == y_train.shape[0], \
            'the size of X_train must equal to the size of y_train'
        assert self.k <= X_train.shape[0], \
            'the size of X_train must be at least k'

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """
        给定待预测的数据集X_predict，返回表示X_predict的结果向量
        """
        assert X_predict is not None and self._y_train is not None, \
            'must fit before predict'
        assert X_predict.shape[0] == self._X_train.shape[1], \
            'the feature number of X_predict must be equal to X_train'

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """
        给定单个的待预测数据集x，返回x的预测结果值
        """
        assert x.shape[0] == self._X_train.shape[1], \
            'the feature number of X_predict must be equal to X_train'

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[neighbor] for neighbor in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k