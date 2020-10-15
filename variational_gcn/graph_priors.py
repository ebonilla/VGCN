# Defines various generative models for graphs
# TODO: in principle, priors should not use tf unless we want to optimize their hyper-paramweters
# but we get troubles if we use numpy now due to how things were coded before


import numpy as np
from scipy.special import expit as sigmoid
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf
from sklearn.neighbors import kneighbors_graph
import tensorflow_probability as tfp


from variational_gcn.utils.base import fill_symmetric, fill_symmetric_inverse
from variational_gcn.math import smooth
from variational_gcn.math import logit


def get_feature_prior(Z):
    probs = tf.to_float(_get_probs_from_logit(_get_logit(Z, type="SE", alpha=-10, lengthscale=1)))
    probs_tril = fill_symmetric_inverse(probs)

    return probs_tril, probs


def _get_logit(Z, type="projection", alpha=0, lengthscale=1):
    if type == "projection":
        Rho = np.matmul(Z, Z.transpose())
    elif type == "distance":
        Rho = - squareform(pdist(Z))
    elif type == "SE":  # Squared exponential
        Rho = np.exp(- np.square(squareform(pdist(Z)) / lengthscale))  # Can have additional length scale parameter
    else:  # need to use a valid distance accepted by pdist
        Rho = - squareform(pdist(Z, type))
        # raise Exception("Invalid type")

    Logit = Rho + alpha
    return Logit


def _get_probs_from_logit(Logit):
    # Generate adjacency matrix as A_{ij} ~ Bernoulli(sigmoid(logit))
    P = sigmoid(Logit)
    return P


def get_smoothing_prior(A, constant, n, one_smoothing_factor, zero_smoothing_factor=1e-5):
    """
    Get array of probabilities corresponding to lower triangular part of adjacency matrix: P = sf*A + (1-sf)*A,
    where sf is the smoothing factor.
    If constant is given, it discards the given adjacency and use the constant instead
    If A is binary, constant=None and smoothing_factor=1 it sinmply return the lower triangular part of A (no probs)
    :param A:
    :param constant:
    :param n:
    :param one_smoothing_factor:
    :param convex_smoothing:
    :param zero_smoothing_factor: float, a value to smoothing the zero entries in adjacency matrix
    :return probs_tril: flat vector of lower triangular probabilities
    :return a: Matrix of probabilities
    """

    a = tf.to_float(A)

    if constant is not None: # discard adjancency (i.e. do not use the graph) and use constant instead
        m = n * (n - 1) // 2 # Number of "free" parameters for adjacency
        a_tril_flat = tf.fill(dims=[m], value=constant)
        a = fill_symmetric(a_tril_flat)

    probs = smooth(a, one_smoothing_factor, zero_smoothing_factor)

    probs_tril = fill_symmetric_inverse(probs)

    return probs_tril, probs


def get_knn_prior(X, k=10, metric="cosine", one_smoothing_factor=0.5, zero_smoothing_factor=1e-5):
    """

    :param X: features
    :param k: number of neighbnours: e.g. [10, 20]
    :param metric: metric, e.g. "cosine", "minkowski"]
    :param one_smoothing_factor:
    :param zero_smoothing_factor
    :return:
    """
    g = kneighbors_graph(X, k, metric=metric)
    A = np.array(g.todense(), dtype=np.float32)

    return get_smoothing_prior(A, constant=None, n=A.shape[0], one_smoothing_factor=one_smoothing_factor,
                               zero_smoothing_factor=zero_smoothing_factor)


def get_free_prior_lowdim(init_size, init_val):
    """
    Gets a free prior over a low-dimensional adjacency to be used for a Kronecker decomposition
    in the likelihood, It actually does not apply any Kronecker operation.
    We don't need to return a tri_l as the parameterization is a a full matrix
    # TODO: There should be a much more efficient way to implement this?
    :param init_size: size of the initiator matrix (aka generator)
    :param init_val: initial values for all entries of initiator
    :param relaxed: True if posterior should be BinaryConcrete distro or Bernoulli otherwise
    :param temperature_prior: Temperature parameter for binary concrete distro
    :return:
    """

    if init_val is None:
        init_val = np.float(1e-5)

    probs = tf.to_float(init_val * np.ones(shape=(init_size, init_size)))

    #probs_flat = tf.reshape(probs, [-1])
    probs_tril = fill_symmetric_inverse(probs)

    return probs_tril, probs


def _get_prior(probs_tril, temperature_prior, relaxed):
    """
    The prior is a product of independent distributions over the entries of the adjacency
    :param probs_tril:
    :param temperature_prior:
    :param relaxed:
    :return:
    """
    # Prior
    if relaxed:  # Define a concrete distributions

        prior = tfp.distributions.Logistic(
            loc=logit(probs_tril) / temperature_prior, scale=1 / temperature_prior
        )

    else:  # Bernoulli distributions

        prior = tfp.distributions.Bernoulli(probs=probs_tril, dtype=tf.float32)

    prior = tfp.distributions.Independent(prior, reinterpreted_batch_ndims=1)

    return prior


def get_prior(prior_type, X, A, constant, n, one_smoothing_factor, zero_smoothing_factor,
              knn, metric, relaxed, temperature_prior, init_size, init_val):
    """

    :param prior_type:
    :param X: Features
    :param A: adjancency matric (if any, can be None)
    :param constant: if all the prior set to a constant
    :param n: Number of nodes in the graph
    :param one_smoothing_factor: Smoothing factor for 1s
    :param zero_smoothing_factor: Smoothing factor for 0s
    :param knn: Number of neighnours in knn graph
    :param relaxed: True if Binary concrete distro to be used
    :param temperature_prior: Temperature parameter of binary concrete distro
    :return:
    """
    if prior_type == "smoothing":
        probs_tril, probs = get_smoothing_prior(A, constant, n, one_smoothing_factor, zero_smoothing_factor)
    elif prior_type == "feature":
        probs_tril, probs = get_feature_prior(X)
    elif prior_type == "knn":
        tf.logging.info("Using KNNG to build prior")
        probs_tril, probs = get_knn_prior(X, k=knn, metric=metric, one_smoothing_factor=one_smoothing_factor,
                                          zero_smoothing_factor=zero_smoothing_factor)
    elif prior_type == "free_lowdim": # for Kronecker augmentation in the likelihood
        probs_flat, probs = get_free_prior_lowdim(init_size=init_size, init_val=init_val)
        probs_tril = probs_flat # flattened matrix of probs
    else:
        raise Exception("Invalid prior type.")
    prior = _get_prior(probs_tril, temperature_prior, relaxed)
    return prior, probs_tril, probs


def get_prior_cluster(
    prior_type,
    X,
    A,
    constant,
    ns,
    one_smoothing_factor,
    zero_smoothing_factor,
    knn,
    metric,
    relaxed,
    temperature_prior,
):
    """

    :param prior_type:
    :param X: Features
    :param A: adjancency matric (if any, can be None)
    :param constant: if all the prior set to a constant
    :param ns: Number of nodes in the graph
    :param one_smoothing_factor: Smoothing factor for 1s
    :param zero_smoothing_factor: Smoothing factor for 0s
    :param knn: Number of neighnours in knn graph
    :param relaxed: True if Binary concrete distro to be used
    :param temperature_prior: Temperature parameter of binary concrete distro
    :return:
    """
    prior = []
    probs_tril = []
    probs = []

    for adj, features, n in zip(A, X, ns):
        if prior_type == "smoothing":
            probs_tril_, probs_ = get_smoothing_prior(
                adj, constant, n, one_smoothing_factor, zero_smoothing_factor
            )
        elif prior_type == "feature":
            probs_tril_, probs_ = get_feature_prior(features)
        elif prior_type == "knn":
            probs_tril_, probs_ = get_knn_prior(
                features,
                k=knn,
                metric=metric,
                one_smoothing_factor=one_smoothing_factor,
                zero_smoothing_factor=zero_smoothing_factor,
            )
        else:
            raise Exception("Invalid prior type.")
        prior_ = _get_prior(probs_tril_, temperature_prior, relaxed)

        prior.append(prior_)
        probs_tril.append(probs_tril_)
        probs.append(probs_)

    return prior, probs_tril, probs
