# -*- coding: utf-8 -*-

"""Metrics module."""

import tensorflow as tf
import tensorflow_probability as tfp


def masked_accuracy(y_true, y_pred, mask):

    correct = tf.to_float(
        tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    )

    return tf.reduce_mean(tf.boolean_mask(correct, mask))


def masked_mnlp(y_true, y_pred_dist, mask):

    return -tf.reduce_mean(tf.boolean_mask(y_pred_dist.log_prob(y_true), mask))


def evaluate_accuracy(y, y_pred, mask_train, mask_val, mask_test):

    matches = tf.to_float(tf.equal(tf.argmax(y, axis=-1), tf.argmax(y_pred, axis=-1)))

    accuracy_train = tf.reduce_mean(tf.boolean_mask(matches, mask_train))
    accuracy_val = tf.reduce_mean(tf.boolean_mask(matches, mask_val))
    accuracy_test = tf.reduce_mean(tf.boolean_mask(matches, mask_test))

    return accuracy_train, accuracy_val, accuracy_test


def evaluate_mnlp(y, y_pred, mask_train, mask_val, mask_test):
    """
    Evaluate the mean negative log predictive probabilities
    :param y:
    :param y_pred:
    :param mask_train:
    :param mask_val:
    :param mask_test:
    :return:
    """
    pred_distro = tfp.distributions.OneHotCategorical(probs=y_pred)
    nlp = -pred_distro.log_prob(y)

    mnlp_train = tf.reduce_mean(tf.boolean_mask(nlp, mask_train))
    mnlp_val = tf.reduce_mean(tf.boolean_mask(nlp, mask_val))
    mnlp_test = tf.reduce_mean(tf.boolean_mask(nlp, mask_test))

    return mnlp_train, mnlp_val, mnlp_test


def evaluate_accuracy_cluster(y, y_pred, mask_train, mask_val, mask_test):

    accuracy_train = []
    accuracy_val = []
    accuracy_test = []

    for y_, y_pred_, mask_train_, mask_val_, mask_test_ in zip(
        y, y_pred, mask_train, mask_val, mask_test
    ):
        matches = tf.to_float(
            tf.equal(tf.argmax(y_, axis=-1), tf.argmax(y_pred_, axis=-1))
        )

        accuracy_train.append(tf.reduce_mean(tf.boolean_mask(matches, mask_train_)))
        accuracy_val.append(tf.reduce_mean(tf.boolean_mask(matches, mask_val_)))
        accuracy_test.append(tf.reduce_mean(tf.boolean_mask(matches, mask_test_)))

    return accuracy_train, accuracy_val, accuracy_test


def evaluate_mnlp_cluster(y, y_pred, mask_train, mask_val, mask_test):
    """
    Evaluate the mean negative log predictive probabilities
    :param y:
    :param y_pred:
    :param mask_train:
    :param mask_val:
    :param mask_test:
    :return:
    """

    mnlp_train = []
    mnlp_val = []
    mnlp_test = []

    for y_, y_pred_, mask_train_, mask_val_, mask_test_ in zip(
        y, y_pred, mask_train, mask_val, mask_test
    ):
        pred_distro = tfp.distributions.OneHotCategorical(probs=y_pred_)
        nlp = -pred_distro.log_prob(y_)

        mnlp_train.append(tf.reduce_mean(tf.boolean_mask(nlp, mask_train_)))
        mnlp_val.append(tf.reduce_mean(tf.boolean_mask(nlp, mask_val_)))
        mnlp_test.append(tf.reduce_mean(tf.boolean_mask(nlp, mask_test_)))

    return mnlp_train, mnlp_val, mnlp_test
