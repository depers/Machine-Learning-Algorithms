# -*- coding: utf-8 -*-
__author__ = 'fengxiao'
__date__ = '2018/2/23 13:18'


def accuracy_socre(y_true, y_predict):
    """
    计算 y_true和 y_predict之间的准确率
    """
    assert y_true.shape[0] == y_predict.shape[0], \
        'the size of y_true must be equal to the size of y_predict'

    return sum(y_true == y_predict) / len(y_true)