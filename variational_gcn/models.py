# -*- coding: utf-8 -*-

"""Convolutional Network Models."""
import warnings

import tensorflow as tf

from tensorflow.keras.layers import Dense

from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Activation

from .preprocessing import (renormalize,
                            laplacian_normalized,
                            laplacian_scaled,
                            chebyshev_terms)

from tensorflow.keras.regularizers import l2

from variational_gcn.layers import LAYER_CLS_NAMES


def basis(x, ms, fn=lambda Phis: tf.concat(Phis, axis=-1)):

    Phis = []

    for m in ms:

        # Shape: (m.shape[0], ..., m.shape[-3], x.shape[-2], x.shape[-1])
        shape = tf.concat([tf.shape(m)[:-2], tf.shape(x)[-2:]], axis=0)

        x = tf.broadcast_to(x, shape=shape)

        Phis.append(tf.matmul(m, x))

    Phi = fn(Phis)

    return Phi


class SpectralGraphConvolutional(Model):

    def __init__(self, output_dim, layer_init_fn=Dense, degree=None, units=16,
                 num_hidden_layers=1, dropout_rate=None,
                 activation='relu', *args, **kwargs):

        super(SpectralGraphConvolutional, self).__init__()

        self.degree = degree

        if dropout_rate is None:
            # identity function. Don't drop any input units.
            self.f = Activation('linear')
        else:
            self.f = Dropout(rate=dropout_rate)

        self.hidden_layers = []

        for i in range(num_hidden_layers):

            layer = layer_init_fn(units, activation=activation, *args, **kwargs)

            self.hidden_layers.append(layer)

        self.output_layer = layer_init_fn(output_dim)

    def call(self, x, a):

        h = self.f(x)

        if self.degree is None:

            ms = [renormalize(a)]

        else:

            l_normalized = laplacian_normalized(a)
            l_scaled = laplacian_scaled(l_normalized)
            ms = list(chebyshev_terms(l_scaled, self.degree))

        # hidden layers
        for layer in self.hidden_layers:

            h = layer(basis(h, ms))
            h = self.f(h)

        # read-out layer
        logits = self.output_layer(basis(h, ms))

        return logits


def build_gcn(degree, output_dim, layer_type, l2_reg_scale_gcn, dropout_rate):
    kwargs = {}

    if layer_type == "dense":

        kwargs["kernel_regularizer"] = l2(0.5 * l2_reg_scale_gcn)
        kwargs["bias_regularizer"] = l2(0.5 * l2_reg_scale_gcn)

    else:

        warnings.warn(
            "kernel_regularizer only available for "
            "layer_type == 'dense'. Ignoring `l2_reg_scale_gcn`."
        )

    gcn = SpectralGraphConvolutional(
        output_dim=output_dim,
        layer_init_fn=LAYER_CLS_NAMES[layer_type],
        degree=degree,
        dropout_rate=dropout_rate,
        **kwargs
    )
    return gcn
