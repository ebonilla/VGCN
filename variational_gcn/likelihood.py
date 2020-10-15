import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from variational_gcn.models import build_gcn
from variational_gcn.math import batch_kronecker_product as batch_tfkron
from variational_gcn.graph_posteriors import sample_posterior


def get_conditional_likelihood_kronecker(x, a_sample, k, degree, dropout_rate, layer_type, l2_reg_scale_gcn,
                                         n, init_size):
    """

    :return:
    """

    print("Unrolling low dim posterior sample using Kronecker product")

    # unroll adjacency sample into Kronecker
    order = np.ceil(np.log(n)/np.log(init_size)).astype(int)  # model order

    a_sample_big = a_sample
    # n_samples = a_sample_big.shape[0].value
    for i in range(order-1):
        a_sample_big = batch_tfkron(a_sample_big, a_sample)

    a_sample_big = a_sample_big[:, :n, :n]     # get's rid of trailing dimensions

    gcn = build_gcn(degree, k, layer_type, l2_reg_scale_gcn, dropout_rate)

    likelihood = tfp.distributions.OneHotCategorical(logits=gcn(x, a_sample_big))

    return likelihood, gcn


def get_conditional_likelihood(
    x, a_sample, k, degree, dropout_rate, layer_type, l2_reg_scale_gcn
):
    """
        Conditional likelihood is a categorical distribution parameterized by a GCN
    :param x:
    :param a_sample:
    :param k: output dimensionality of gcn
    :param degree:
    :param dropout_rate:
    :param layer_type:
    :param l2_reg_scale_gcn:
    :return:
    """

    gcn = build_gcn(degree, k, layer_type, l2_reg_scale_gcn, dropout_rate)

    likelihood = tfp.distributions.OneHotCategorical(logits=gcn(x, a_sample))

    return likelihood, gcn


def predict(likelihood):
    """
    Make prediction using current likelihood model, parameters and samples from posterior
    :param likelihood:
    :return: (n,k) matrix of predictive probabilities
    """
    # shape (mc_sample_size, n, k)
    # y_pred_samples = likelihood.mode()
    y_pred = tf.reduce_mean(likelihood.probs, axis=0)

    # shape (n, k)
    return y_pred


def predict_cluster(likelihood):
    """
    Make prediction using current likelihood model, parameters and samples from posterior
    :param likelihood:
    :return: (n,k) matrix of predictive probabilities
    """
    y_pred = []
    for ll in likelihood:
        y_pred.append(tf.reduce_mean(ll.probs, axis=0))

    # shape (n, k)
    return y_pred

def get_predictive_distribution(x, gcn, posterior,  relaxed, mc_samples_test):
    """
    Gets predictive probabilities
    :param gcn:
    :param posterior:
    :param relaxed:
    :param mc_samples_test:
    :return:
    """
    a_sample, _, _ = sample_posterior(posterior, relaxed, mc_samples_test)
    likelihood = tfp.distributions.OneHotCategorical(logits=gcn(x, a_sample))
    y_pred = predict(likelihood)
    return y_pred


def get_predictive_distribution_cluster(x, gcn, posterior, relaxed, mc_samples_test):
    """
    Gets predictive probabilities
    :param gcn:
    :param posterior:
    :param relaxed:
    :param mc_samples_test:
    :return:
    """
    y_pred = []

    for posterior_, x_ in zip(posterior, x):
        a_sample, _, _ = sample_posterior(posterior_, relaxed, mc_samples_test)
        likelihood = tfp.distributions.OneHotCategorical(logits=gcn(x_, a_sample))
        y_pred.append(predict(likelihood))

    return y_pred

def get_conditional_likelihood_cluster(
    x, a_sample, k, degree, dropout_rate, layer_type, l2_reg_scale_gcn
):
    """
        Conditional likelihood is a categorical distribution parameterized by a GCN
    :param x:
    :param a_sample:
    :param k: output dimensionality of gcn
    :param degree:
    :param dropout_rate:
    :param layer_type:
    :param l2_reg_scale_gcn:
    :return:
    """

    gcn = build_gcn(degree, k, layer_type, l2_reg_scale_gcn, dropout_rate)

    likelihood = []
    for a_sample_, x_ in zip(a_sample, x):
        likelihood.append(
            tfp.distributions.OneHotCategorical(logits=gcn(x_, a_sample_))
        )

    return likelihood, gcn