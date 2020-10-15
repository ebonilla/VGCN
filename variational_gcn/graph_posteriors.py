# TODO: In principle, we don't need to distinguish priors from posteriors and have graph distributions
# In practice they're different since we need to optimize posteriors (i.e. variables in tensorflow)

import tensorflow as tf
import tensorflow_probability as tfp
from variational_gcn.models import SpectralGraphConvolutional
from variational_gcn.utils.base import fill_symmetric, fill_symmetric_inverse
from variational_gcn.math import logit
import scipy as sp
import numpy as np

from variational_gcn.math import kronecker_product as tfkron


def get_distro_from_logits(logits, relaxed, temperature):

    if relaxed:  # Concrete case
        dist = tfp.distributions.Logistic(loc=logits / temperature, scale=1 / temperature)
    else:  # Discrete distro
        dist = tf.distributions.Bernoulli(logits=logits, dtype=tf.float32)

    distro = tfp.distributions.Independent(dist, reinterpreted_batch_ndims=1)

    return distro


def get_amortized_posterior(x, a, relaxed, temperature_posterior, latent_dim, logit_shift):

    def _make_posterior(fn, logit_shift, temperature=None):
        def conditional(x, a):

            z = fn(x, a)

            K = tf.matmul(z, tf.transpose(z))
            logits_tril = fill_symmetric_inverse(K)
            logits = logits_tril - logit_shift

            distro = get_distro_from_logits(logits, relaxed, temperature_posterior)

            return distro

        return conditional

    posterior_param = None
    # TODO: Make these arguments available on command-line.
    encoder_gcn = SpectralGraphConvolutional(output_dim=latent_dim, num_hidden_layers=2, dropout_rate=None, units=128)
    if relaxed:
        posterior_fn = _make_posterior(encoder_gcn, logit_shift=logit_shift, temperature=temperature_posterior)
    else:
        posterior_fn = _make_posterior(encoder_gcn, logit_shift=logit_shift, temperature=None)

    posterior = posterior_fn(x, a)

    return posterior, posterior_param


def get_free_posterior(relaxed, temperature_posterior, probs_tril, name="log_alpha"):

    log_alpha = tf.get_variable(name=name, initializer=logit(probs_tril))
    posterior = get_distro_from_logits(log_alpha, relaxed, temperature_posterior)
    posterior_param = log_alpha

    return posterior, posterior_param


def get_lowrank_posterior(n, latent_dim, logit_shift, a, relaxed, temperature, use_graph):

    def _init_posterior_low_rank(n, d, a, logit_shift, use_graph):
        """
        initialize low-rank parameterization of matrix of probabilties
        :param n:
        :param d: number of latent dimensions
        :param a:
        :param logit_shift:
        :param use_graph:
        :return:
        """
        # TODO: include offset due to other shifting variables z_i and z_j
        if use_graph is False:  # a is matrix of constants
            init_prob = tf.reduce_mean(a)
            val = logit(init_prob) + logit_shift
            z = tf.sqrt(val / d) * tf.ones(shape=[n, d])
        else:  # a is matrix of probabilities
            # if not isinstance(a, np.ndarray): # a is a tensor
            #     a_np = tf.Session().run(a)  # need to resort to numpy to use eigs (yuck!)
            # else:
            #     a_np = a
            a_np = tf.Session().run(a)  # need to resort to numpy to use eigs (yuck!)
            val = sp.special.logit(a_np) + logit_shift

            np.fill_diagonal(val, 1)

            #eig_vals, eig_vecs = sp.sparse.linalg.eigs(val, k=d)
            #eig_vals = np.real(eig_vals)
            #eig_vecs = np.real(eig_vecs)
            #
            #eig_vals[eig_vals < 0] = 0
            #z = np.real(eig_vecs) * np.sqrt(eig_vals)

            u, s, vt = sp.sparse.linalg.svds(val, k=d)

            # print(z)
        return u, s, vt

    u0, s0, vt0 = _init_posterior_low_rank(n, latent_dim, a, logit_shift, use_graph)

    u = tf.get_variable(name="u", initializer=u0 * s0)
    # s = tf.get_variable(name="s", initializer=s0)
    vt = tf.get_variable(name="vt", initializer=vt0)

    b_z = tf.get_variable(name="bias_z", shape=[n], initializer=tf.zeros_initializer)
    shift_z = tf.get_variable(name="shift_z", initializer=logit_shift)

    #K = tf.matmul(z, tf.transpose(z))
    #K = tf.matmul(z * l, tf.transpose(z))

    # K = tf.matmul(u * s, vt)
    K = tf.matmul(u, vt)

    # K(i,j) = K(i,j) + b_z(i) + b_z(j)
    K = K + tf.expand_dims(b_z, axis=0) + tf.expand_dims(b_z, axis=1)

    logits_tril = fill_symmetric_inverse(K)

    logits = logits_tril - shift_z

    posterior = get_distro_from_logits(logits, relaxed, temperature)

    #posterior_param = [z, b_z, shift_z]
    posterior_param = [u,  vt]

    return posterior, posterior_param


def get_free_posterior_lowdim(init_size, init_val, relaxed, temperature_posterior):
    """
    Gets a free posterior over a low-dimensional adjacency to be used for a Kronecker decomposition
    in the likelihood, It actually does not apply any Kronecker operation
    # TODO: There should be a much more efficient way to implement this?
    :param init_size: size of the initiator matrix (aka generator)
    :param init_val: initial values for all entries of initiator
    :param relaxed: True if posterior should be BinaryConcrete distro or Bernoulli otherwise
    :param temperature_posterior: Temperature parameter for binary concrete distro
    :return:
    """

    if init_val is None:
        init_val = np.float(1e-5)

    K = tf.get_variable(name="initK", initializer=init_val * tf.ones(shape=(init_size, init_size)))
    logits = logit(K)

    # TODO: I AM HERE: Do I need to constraint my generator to be symmetric?
    # Can I learn an undirected graph?
    logits_tril = tf.reshape(logits, [-1]) # flattened version of probabilities
    posterior = get_distro_from_logits(logits_tril, relaxed, temperature_posterior)
    posterior_param = [K]

    return posterior, posterior_param


def get_kronecker_posterior(n, init_size, init_val, relaxed, temperature_posterior):
    """
    Gets a kronecker product posterior
    # TODO: There should be a much more efficient way to implement this?
    :param n: Number of nodes in the graph
    :param init_size: size of the initiator matrix (aka generator)
    :param init_val: initial values for all entries of initiator
    :param relaxed: True if posterior should be BinaryConcrete distro or Bernoulli otherwise
    :param temperature_posterior: Temperature parameter for binary concrete distro
    :return:
    """

    order = np.ceil(np.log(n)/np.log(init_size)).astype(int)  # model order
    # print(order)
    if init_val is None:
        init_val = np.power(1e-5, 1.0/order)

    # print(init_val)
    initK = tf.get_variable(name="initK", initializer=init_val * tf.ones(shape=(init_size, init_size)))
    K = initK
    for i in range(order-1):
        K = tfkron(K, initK)

    K = K[:n, :n]     # get's rid of trailing dimensions

    logits = logit(K)
    logits_tril = fill_symmetric_inverse(logits)
    posterior = get_distro_from_logits(logits_tril, relaxed, temperature_posterior)
    posterior_param = [initK]

    return posterior, posterior_param


def sample_posterior(posterior, relaxed, mc_sample_size):
    """
    Sample from variational posterior
    :param posterior:
    :param relaxed:
    :param mc_sample_size:
    :return:
    """
    b_sample_tril = None

    if relaxed:
        b_sample_tril = posterior.sample(mc_sample_size)
        a_sample_tril = tf.sigmoid(b_sample_tril)
    else:
        a_sample_tril = posterior.sample(mc_sample_size)

    a_sample = fill_symmetric(a_sample_tril)

    return a_sample, a_sample_tril, b_sample_tril


def get_variational_posterior(
    relaxed,
    posterior_type,
    temperature_posterior,
    x,
    a,
    probs_tril,
    n,
    latent_dim,
    logit_shift,
    use_graph
):
    """
    Variational posterior is a product of indepedent distributions over the adjacency entries
    :param relaxed:
    :param amortized:
    :param temperature_posterior:
    :param reg_scale_l1:
    :param reg_scale_l2:
    :param x:
    :param a:
    :param probs_tril:
    :return:
    """
    if posterior_type == "amortized":
        posterior, posterior_param = get_amortized_posterior(x, a, relaxed, temperature_posterior,
                                                             latent_dim, logit_shift)

    elif posterior_type == "free":
        posterior, posterior_param = get_free_posterior(relaxed, temperature_posterior, probs_tril)

    elif posterior_type == "lowrank":
        posterior, posterior_param = get_lowrank_posterior(n, latent_dim, logit_shift, a, relaxed,
                                                           temperature_posterior, use_graph)
    elif posterior_type == "kronecker":
        posterior, posterior_param = get_kronecker_posterior(n, init_size=4, init_val=None, relaxed=relaxed,
                                                             temperature_posterior=temperature_posterior)
    elif posterior_type == "free_lowdim":
        # posterior, posterior_param = get_free_posterior_lowdim(init_size=init_size, init_val=init_val,
        #                                                       relaxed=relaxed,
        #                                                      temperature_posterior=temperature_posterior)
        posterior, posterior_param = get_free_posterior(relaxed, temperature_posterior, probs_tril)
    else:
        raise Exception("Invalid posterior type")

    return posterior, posterior_param


def get_variational_posterior_cluster(
    relaxed,
    posterior_type,
    temperature_posterior,
    x,
    a,
    probs_tril,
    ns,
    latent_dim,
    logit_shift,
    use_graph,
):
    """
    Variational posterior is a product of indepedent distributions over the adjacency entries
    :param relaxed:
    :param amortized:
    :param temperature_posterior:
    :param reg_scale_l1:
    :param reg_scale_l2:
    :param x:
    :param a:
    :param probs_tril:
    :return:
    """

    posterior = []
    posterior_param = []
    for indx, (probs_, probs_tril_, x_, n) in enumerate(zip(a, probs_tril, x, ns)):
        if posterior_type == "amortized":
            posterior_, posterior_param_ = get_amortized_posterior(
                x_, probs_, relaxed, temperature_posterior, latent_dim, logit_shift
            )

        elif posterior_type == "free":
            posterior_, posterior_param_ = get_free_posterior(
                relaxed,
                temperature_posterior,
                probs_tril_,
                name="log_alpha_" + str(indx),
            )

        elif posterior_type == "lowrank":
            posterior_, posterior_param_ = get_lowrank_posterior(
                n,
                latent_dim,
                logit_shift,
                probs_,
                relaxed,
                temperature_posterior,
                use_graph,
            )
        elif posterior_type == "kronecker":
            posterior_, posterior_param_ = get_kronecker_posterior(
                n,
                init_size=4,
                init_val=None,
                relaxed=relaxed,
                temperature_posterior=temperature_posterior,
            )
        else:
            raise Exception("Invalid posterior type")

        posterior.append(posterior_)
        posterior_param.append(posterior_param_)

    return posterior, posterior_param