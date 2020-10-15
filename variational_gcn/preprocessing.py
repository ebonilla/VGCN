# -*- coding: utf-8 -*-

"""Preprocessing functions."""

import tensorflow as tf


def eye_like(a, num_columns=None, dtype=tf.float32, name=None):

    batch_shape = tf.shape(a)[:-2]
    num_rows = tf.shape(a)[-1]

    return tf.eye(num_rows=num_rows, num_columns=num_rows,
                  batch_shape=batch_shape, dtype=tf.float32, name=None)


def adjacency_normalized(a):

    d_diag = tf.reduce_sum(a, axis=-1)
    d_inv_sqrt_diag = tf.pow(d_diag, -0.5)
    d_inv_sqrt = tf.matrix_diag(d_inv_sqrt_diag)

    # TODO(LT): Since `a` is symmetric and `d_inv_sqrt` is diagonal,
    # shouldn't they commute?
    return tf.matmul(tf.matmul(d_inv_sqrt, a), d_inv_sqrt)


def renormalize(a):
    """
    Implements the "renormalization trick".
    """
    a_tilde = a + eye_like(a)

    return adjacency_normalized(a_tilde)


def laplacian_normalized(a):

    return eye_like(a) - adjacency_normalized(a)


def eigvalsh_largest(l):
    """
    Computes the *largest* eigenvalue of self-adjoint matrices.
    """

    eigvalsh_l = tf.linalg.eigvalsh(l)

    return eigvalsh_l[..., -1]


def laplacian_scaled(l):

    l_eigvalsh_largest = eigvalsh_largest(l)

    # reshape (...) to (..., 1, 1) for broadcasting
    l_eigvalsh_largest = tf.expand_dims(l_eigvalsh_largest, axis=-1)
    l_eigvalsh_largest = tf.expand_dims(l_eigvalsh_largest, axis=-1)

    return 2. * tf.truediv(l, l_eigvalsh_largest) - eye_like(l)


def chebyshev_terms(X, k):

    A, B = eye_like(X), X

    for _ in range(k+1):

        yield A

        A, B = B, 2. * X @ B - A
