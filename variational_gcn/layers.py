# -*- coding: utf-8 -*-

"""Layers. **Deprecated**."""

import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense

LAYER_CLS_NAMES = dict(
    dense=Dense,
    dense_flipout=tfp.layers.DenseFlipout,
    dense_reparam=tfp.layers.DenseReparameterization,
    dense_reparam_local=tfp.layers.DenseLocalReparameterization
)
