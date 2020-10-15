import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from variational_gcn.math import reduce_logmeanexp


def get_kl_divergence_discrete(prior, posterior):
    # shape (,)
    kl = tfp.distributions.kl_divergence(posterior, prior)
    return kl


def get_ell_discrete(
    posterior, a_sample_tril, likelihood, y, mask_train, mask_val, mask_test
):
    """
         Computes the expectation of the conditional likelihood over the current posterior distribution
    :param posterior:
    :param a_sample_tril:
    :param likelihood:
    :param y:
    :param relaxed:
    :param mask_train:
    :param mask_val:
    :param mask_test:
    :return:
    """

    # shape (mc_sample_size, n)
    ell_local = likelihood.log_prob(y)

    q_log_prob = lambda x: tf.expand_dims(posterior.log_prob(x), axis=-1)
    ell_all = tfp.monte_carlo.expectation(
        f=lambda _: ell_local,
        samples=a_sample_tril,
        log_prob=q_log_prob,
        use_reparametrization=False,
        axis=0,
    )

    ell_train = tf.reduce_sum(tf.boolean_mask(ell_all, mask_train))
    ell_val = tf.reduce_sum(tf.boolean_mask(ell_all, mask_val))
    ell_test = tf.reduce_sum(tf.boolean_mask(ell_all, mask_test))

    return ell_train, ell_val, ell_test


def get_losses(
    prior,
    posterior,
    likelihood,
    y,
    a_sample_tril,
    b_sample_tril,
    gcn,
    mask_train,
    mask_val,
    mask_test,
    relaxed,
    beta=1.0
):
    n_train = np.sum(mask_train)
    n_val = np.sum(mask_val)
    n_test = np.sum(mask_test)

    if relaxed is True:
        elbo_train, elbo_val, elbo_test, kl, ell_train = get_elbo_relaxed(
            prior,
            posterior,
            likelihood,
            y,
            b_sample_tril,
            mask_train,
            mask_val,
            mask_test,
            beta
        )
    else:
        elbo_train, elbo_val, elbo_test, kl, ell_train = get_elbo_discrete(
            prior,
            posterior,
            likelihood,
            y,
            a_sample_tril,
            mask_train,
            mask_val,
            mask_test,
            beta
        )

    # Report average elbo and losses (for consistency with GCNs)
    elbo_train = elbo_train / n_train
    elbo_val = elbo_val / n_val
    elbo_test = elbo_test / n_test
    kl = kl / n_train
    ell_train = ell_train / n_train

    reg_log_alpha = tf.losses.get_regularization_loss()
    reg_gcn = tf.reduce_sum(gcn.losses)

    reg = reg_log_alpha + reg_gcn

    loss_train = -elbo_train + reg
    loss_val = -elbo_val + reg
    loss_test = -elbo_test + reg

    return (
        elbo_train,
        elbo_val,
        elbo_test,
        loss_train,
        loss_val,
        loss_test,
        reg,
        kl,
        ell_train,
    )


def get_elbo_relaxed(
    prior, posterior, likelihood, y, b_sample_tril, mask_train, mask_val, mask_test, beta=1.0
):
    n_train = np.sum(mask_train)
    n_val = np.sum(mask_val)
    n_test = np.sum(mask_test)

    # shape (mc_sample_size, n)
    ell_local = likelihood.log_prob(y)
    kl_local = beta * tf.expand_dims(
        posterior.log_prob(b_sample_tril) - prior.log_prob(b_sample_tril), axis=-1
    )

    elbo_local_train = reduce_logmeanexp(ell_local - kl_local / n_train, axis=0)
    elbo_local_val = reduce_logmeanexp(ell_local - kl_local / n_val, axis=0)
    elbo_local_test = reduce_logmeanexp(ell_local - kl_local / n_test, axis=0)

    elbo_train = tf.reduce_sum(tf.boolean_mask(elbo_local_train, mask_train))
    elbo_val = tf.reduce_sum(tf.boolean_mask(elbo_local_val, mask_val))
    elbo_test = tf.reduce_sum(tf.boolean_mask(elbo_local_test, mask_test))

    # individual kl, ell for reporting purposes. Not quite the same used to compute the loss since this one
    # is done using the IWELBO (logmeanexp non-separable)
    kl = tf.reduce_sum(
        reduce_logmeanexp(kl_local, axis=0)
    )  # Outer sum is doing nothing

    ell_train = elbo_train + kl

    return elbo_train, elbo_val, elbo_test, kl, ell_train


def get_elbo_discrete(
    prior, posterior, likelihood, y, a_sample_tril, mask_train, mask_val, mask_test, beta=1.0
):

    kl = beta * get_kl_divergence_discrete(prior, posterior)
    ell_train, ell_val, ell_test = get_ell_discrete(
        posterior, a_sample_tril, likelihood, y, mask_train, mask_val, mask_test
    )

    elbo_train = ell_train - kl
    elbo_val = ell_val - kl
    elbo_test = ell_test - kl

    return elbo_train, elbo_val, elbo_test, kl, ell_train

def get_losses_cluster(
    prior,
    posterior,
    likelihood,
    y,
    a_sample_tril,
    b_sample_tril,
    gcn,
    mask_train,
    mask_val,
    mask_test,
    relaxed,
    beta=1.0,
):
    elbo_train = []
    elbo_val = []
    elbo_test = []
    loss_train = []
    loss_val = []
    loss_test = []
    reg = []
    kl = []
    ell_train = []

    for (
        prior_,
        posterior_,
        likelihood_,
        y_,
        a_sample_tril_,
        b_sample_tril_,
        mask_train_,
        mask_val_,
        mask_test_,
    ) in zip(
        prior,
        posterior,
        likelihood,
        y,
        a_sample_tril,
        b_sample_tril,
        mask_train,
        mask_val,
        mask_test,
    ):

        n_train = np.sum(mask_train_)
        n_val = np.sum(mask_val_)
        n_test = np.sum(mask_test_)

        if relaxed is True:
            elbo_train_, elbo_val_, elbo_test_, kl_, ell_train_ = get_elbo_relaxed(
                prior_,
                posterior_,
                likelihood_,
                y_,
                b_sample_tril_,
                mask_train_,
                mask_val_,
                mask_test_,
                beta,
            )
        else:
            elbo_train_, elbo_val_, elbo_test_, kl_, ell_train_ = get_elbo_discrete(
                prior_,
                posterior_,
                likelihood_,
                y_,
                a_sample_tril_,
                mask_train_,
                mask_val_,
                mask_test_,
                beta,
            )

        # Report average elbo and losses (for consistency with GCNs)
        elbo_train_ = elbo_train_ / n_train
        elbo_val_ = elbo_val_ / n_val
        elbo_test_ = elbo_test_ / n_test
        kl_ = kl_ / n_train
        ell_train_ = ell_train_ / n_train

        # TODO: Check that the below is doing the correct calculation
        reg_log_alpha = tf.losses.get_regularization_loss()
        reg_gcn = tf.reduce_sum(gcn.losses)

        reg_ = reg_log_alpha + reg_gcn

        loss_train_ = -elbo_train_ + reg_
        loss_val_ = -elbo_val_ + reg_
        loss_test_ = -elbo_test_ + reg_

        elbo_train.append(elbo_train_)
        elbo_val.append(elbo_val_)
        elbo_test.append(elbo_test_)
        loss_train.append(loss_train_),
        loss_val.append(loss_val_)
        loss_test.append(loss_test_)
        reg.append(reg_)
        kl.append(kl_)
        ell_train.append(ell_train_)

    return (
        elbo_train,
        elbo_val,
        elbo_test,
        loss_train,
        loss_val,
        loss_test,
        reg,
        kl,
        ell_train,
    )
