# -*- coding: utf-8 -*-

"""Helper functions and classes."""

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import scipy.sparse as sps
import networkx as nx

import random
import string

from sklearn.model_selection import StratifiedShuffleSplit


def sparse_to_tuple(m):

    if not sps.isspmatrix_coo(m):
        m = m.tocoo()

    indices = np.vstack((m.row, m.col)).transpose()
    values = np.float32(m.data)
    dense_shape = m.shape

    return indices, values, dense_shape


def recursive_stratified_shuffle_split(sizes, random_state=None):
    """
    """
    head, *tail = sizes
    sss = StratifiedShuffleSplit(n_splits=1, test_size=head,
                                 random_state=random_state)

    def split(X, y):

        a_index, b_index = next(sss.split(X, y))

        yield a_index

        if tail:

            split_tail = recursive_stratified_shuffle_split(sizes=tail,
                                                            random_state=random_state)

            for ind in split_tail(X[b_index], y[b_index]):

                yield b_index[ind]

        else:

            yield b_index

    return split


def indices_to_mask(indices, size):

    mask = np.zeros(size, dtype=np.bool)
    mask[indices] = True

    return mask


def mask_values(a, mask, fill_value=0):

    a_masked = np.full_like(a, fill_value, dtype=np.int32)
    a_masked[mask] = a[mask]

    return a_masked


def fill_symmetric(tril_flat):

    batch_rank = tf.rank(tril_flat) - 1

    lower = tfp.distributions.fill_triangular(tril_flat)

    paddings = tf.concat([tf.zeros([batch_rank, 2], dtype=tf.int32),
                          tf.constant([[1, 0], [0, 1]])], axis=0)

    lower = tf.pad(lower, paddings=paddings)

    perm = tf.concat([tf.range(batch_rank), [batch_rank + 1, batch_rank]],
                     axis=0)

    upper = tf.transpose(lower, perm=perm)

    a = lower + upper

    return a


def fill_symmetric_inverse(a):

    tril = tf.linalg.band_part(a[1:, :-1], -1, 0)
    tril_flat = tfp.distributions.fill_triangular_inverse(tril)

    return tril_flat


def remove_links_from_adj(a, perc=0.1):
    '''
    Removes, uniformly at random, perc of edges from the adjacency matrix a.
    :param a: The adjacency matrix (assumed binary).
    :param perc: The fraction of edges to corrupt (as a fraction of the number of edges in the graph.)
    :return: The corrupted adjacency matrix with links removed.
    '''
    au = np.triu(a, k=1)
    r, c = np.where(au > 0.0)  # find the locations in a with edges
    print("len(r): {}".format(len(r)))
    idx = np.random.choice(len(r), size=int(len(r) * perc), replace=False)
    a[r[idx], c[idx]] = 0
    a[c[idx], r[idx]] = 0

    return a


def add_links_to_adj(a, perc=0.1, n=None):
    '''
    It switches 0s in the adjacency matrix to 1s essentially adding edges where no edges
    exist in the graph.
    :param a: The adjacency matrix (assumed binary).
    :param perc: The fraction of edges to corrupt (as a fraction of the number of edges in the graph).
    :return: The corrupted adjacency matrix with links added.
    '''
    au = np.triu(a, k=1)
    r, c = np.where(au < 0.5)  # find the locations in a without edges

    re, ce = np.where(au >= 0.5)  # find the locations in a with edges

    print("len(r): {}".format(len(r)))
    num_to_add = n
    if num_to_add is None:
        num_to_add = int(len(re) * perc)

    idx = np.random.choice(len(r), size=num_to_add, replace=False)

    a[r[idx], c[idx]] = 1
    a[c[idx], r[idx]] = 1

    return a


def noisy_adj(a, perc=0.1):
    """
    It removes perc number of edges and adds perc number of edges. The total number of
    edges in the graph remain the same.
    :param a: The ajacency matrix (assumed binary)
    :param perc: The fraction of edges to corrupt (as a fraction of the number of edges in the graph.)
    :return: The corrupted adjacency matrix.
    """

    au = np.triu(a, k=1)

    r, c = np.where(au > 0)
    print("len(r): {}".format(len(r)))

    num_to_flip = int(0.5*len(r)*perc)

    idx = np.random.choice(len(r), size=num_to_flip, replace=False)
    a[r[idx], c[idx]] = 0
    a[c[idx], r[idx]] = 0

    r, c = np.where(au < 1)

    idx = np.random.choice(len(r), size=num_to_flip, replace=False)
    a[r[idx], c[idx]] = 1
    a[c[idx], r[idx]] = 1

    return a


def corrupt_adjacency(A, adj_corruption_method, perc_corruption, adjacency_matrix=None):
    if adjacency_matrix is not None:
        g_ = nx.read_gpickle(adjacency_matrix)
        A = nx.adjacency_matrix(g_).toarray()
    elif adj_corruption_method is not None:
        if adj_corruption_method == "noisy":
            A = noisy_adj(A, perc_corruption)
            print("Corrupting with noise perc links {}".format(perc_corruption))
        elif adj_corruption_method == "missing":
            A = remove_links_from_adj(A, perc_corruption)
            print("Corrupting with missing links, perc: {}".format(perc_corruption))
        elif adj_corruption_method == "adding":
            A = add_links_to_adj(A, perc_corruption)
            print("Corrupting with adding links, perc: {}".format(perc_corruption))
    return A


def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    """
    generates a random sequence of character of length ``size``
    """

    return ''.join(random.choice(chars) for _ in range(size))


def get_experiment_ID():
    """
    :return: A random ID
    """
    return id_generator(size=6)


def split_train_val(mask_train, mask_val, seed_val):
    # Split the val set to two equal halves and then add one to the training set and leave the other half
    # for validation
    # The below code is similar to the LDS method reorganize_data_for_es in lds_gnn.data
    rnd_val = np.random.RandomState(seed_val)
    p = 0.5
    chs = rnd_val.choice([True, False], size=np.sum(mask_val), p=[p, 1.0 - p])
    mask_val_new = np.array(mask_val)
    mask_train_new = np.array(mask_train)
    mask_val_new[mask_val_new] = chs
    mask_train_new[mask_val] = ~chs

    # mask_val = mask_val_new
    # mask_train = mask_train_new


    print(
        "**********************************************************************************************"
    )
    print(
        "Using some of the validation data in training: train size: {} val size: {} ".format(
            np.sum(mask_train_new), np.sum(mask_val_new)
        )
    )
    print(
        "**********************************************************************************************"
    )

    return mask_train_new, mask_val_new

