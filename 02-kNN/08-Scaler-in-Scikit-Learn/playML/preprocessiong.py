# -*- coding: utf-8 -*-
import numpy as np

__author__ = 'fengxiao'
__date__ = '2018/3/18 15:22'


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        根据训练数据集X获得数据的均值方差
        """
        assert X.ndim == 2, 'The dimension of X must be 2.'

        self.mean_ = np.array([np.mean((X[:i]) for i in range(X.shape[1]))])

        return self

    def transform(self, X):
        """
        将X根据这个StandardScaler进行均值方差的归一化处理
        """
        assert X.ndim == 2, 'The dimension of X must be 2.'
        assert self.mean_ is not None and self.scale_ is not None, \
            'must fit before transform.'
        assert X.shape[1] == len(self.mean_), \
            'the feature number of X must be equal to mean_ and std_'

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:col] = (X[:col] - self.mean_) / self.scale_[col]

        return resX
