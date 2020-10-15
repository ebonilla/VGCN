# -*- coding: utf-8 -*-

"""Math."""

import tensorflow as tf


def logit(p):
    """
    The logit, or log-odds function. Inverse of the logistic sigmoid function.
    """
    return tf.log(p) - tf.log1p(-p)


def smooth(x, one_smoothing_factor, zero_smoothing_factor=1e-5):
    """
    :param x:
    :param one_smoothing_factor:
    :param zero_smoothing_factor: float smoothing factor for the zero entries in adjacency matrix
    :return:
    """

    if one_smoothing_factor < 0:
        return abs(one_smoothing_factor) * x
    else:
        return one_smoothing_factor * x + zero_smoothing_factor * (1. - x)


def reduce_logmeanexp(x, axis):

    return (tf.reduce_logsumexp(x, axis=axis) -
            tf.log(tf.to_float(tf.shape(x)[axis])))


def kronecker_product(mat1, mat2):
    """
        Computes the Kronecker product two matrices.
        https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/utils.py
    :param mat1:
    :param mat2:
    :return:
    """
    m1, n1 = mat1.get_shape().as_list()
    mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
    m2, n2 = mat2.get_shape().as_list()
    mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
    return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def batch_kronecker_product(mat1, mat2):
    s1, m1, n1 = mat1.get_shape().as_list()
    mat1_rsh = tf.reshape(mat1, [s1, m1, 1, n1, 1])
    s2, m2, n2 = mat2.get_shape().as_list()
    mat2_rsh = tf.reshape(mat2, [s2, 1, m2, 1, n2])
    return tf.reshape(mat1_rsh * mat2_rsh, [s1, m1 * m2, n1 * n2])
