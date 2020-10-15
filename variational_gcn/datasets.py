# -*- coding: utf-8 -*-

"""Dataset-loading utilities module."""

import os.path

import numpy as np
import scipy.sparse as sps

import pandas as pd
import networkx as nx
import pickle as pkl

from functools import partial

from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_iris as _load_iris
from scipy.sparse import coo_matrix

from sklearn.preprocessing import normalize

from variational_gcn.utils.base import (
    recursive_stratified_shuffle_split,
    indices_to_mask,
    mask_values,
)

import tensorflow as tf


def load_pickle(name, ext, data_home="datasets", encoding="latin1"):

    path = os.path.join(data_home, name, "ind.{0}.{1}".format(name, ext))

    with open(path, "rb") as f:

        return pkl.load(f, encoding=encoding)


def load_test_indices(name, data_home="datasets"):

    indices_df = pd.read_csv(
        os.path.join(data_home, name, "ind.{0}.test.index".format(name)), header=None
    )
    indices = indices_df.values.squeeze()

    return indices


def load_dataset(name, data_home="datasets"):

    exts = ["tx", "ty", "allx", "ally", "graph"]

    (X_test, y_test, X_rest, y_rest, G_dict) = map(
        partial(load_pickle, name, data_home=data_home), exts
    )

    _, D = X_test.shape
    _, K = y_test.shape

    ind_test_perm = load_test_indices(name, data_home)
    ind_test = np.sort(ind_test_perm)

    num_test = len(ind_test)
    num_test_full = ind_test[-1] - ind_test[0] + 1

    # TODO: Issue warning if `num_isolated` is non-zero.
    num_isolated = num_test_full - num_test

    # normalized zero-based indices
    ind_test_norm = ind_test - np.min(ind_test)

    # features
    X_test_full = sps.lil_matrix((num_test_full, D))
    X_test_full[ind_test_norm] = X_test

    X_all = sps.vstack((X_rest, X_test_full)).toarray()
    X_all[ind_test_perm] = X_all[ind_test]

    # targets
    y_test_full = np.zeros((num_test_full, K))
    y_test_full[ind_test_norm] = y_test

    y_all = np.vstack((y_rest, y_test_full))
    y_all[ind_test_perm] = y_all[ind_test]

    # graph
    G = nx.from_dict_of_lists(G_dict)
    A = nx.to_scipy_sparse_matrix(G, format="coo")

    return (X_all, y_all, A)


def load_split(name, val_size=500, data_home="datasets"):

    y = load_pickle(name, ext="y", data_home=data_home)

    train_size = len(y)

    ind_train = range(train_size)
    ind_val = range(train_size, train_size + val_size)

    ind_test_perm = load_test_indices(name, data_home)
    ind_test = np.sort(ind_test_perm)

    return ind_train, ind_val, ind_test


def load_iris():

    X_all, y = _load_iris(return_X_y=True)
    y_all = LabelBinarizer().fit_transform(y)

    n, d = X_all.shape

    # no graph structure
    A = coo_matrix((n, n))

    return (X_all, y_all, A)


def load_karate_club():

    G = nx.karate_club_graph()

    A = nx.to_scipy_sparse_matrix(G, format="coo")

    # no features
    X_all = np.eye(G.number_of_nodes())

    labels = [G.node[n]["club"] for n in G.nodes()]
    y_all = LabelBinarizer().fit_transform(labels)

    return (X_all, y_all, A)


def load_twitter(data_home="datasets/twitter"):

    # Load the node features. The index is node IDs
    df = pd.read_csv(
        os.path.join(data_home, "twitter_node_features.csv"), header=0, index_col=0
    )

    print("Loaded node features {}".format(df.shape))
    # Extract the target values

    df_y = pd.DataFrame({"hate": df["hate"], "not-hate": 1 - df["hate"]})

    y_all = df_y.values
    # y_all = LabelBinarizer().fit_transform(y_all)
    # Drop the column with the target values
    df.drop(columns=["hate"], inplace=True)

    X_all = df.values

    print("y_all shape {}".format(y_all.shape))
    print("X_all shape {}".format(X_all.shape))

    # Load the graph
    print("Loading graph")
    g_nx = nx.read_adjlist(os.path.join(data_home, "twitter.edges"), nodetype=int)
    print("...Done")
    print(
        "Graph num nodes {} and edges {}".format(
            g_nx.number_of_nodes(), g_nx.number_of_edges()
        )
    )
    # Extract the adjacency matrix.
    # A should be scipy sparse matrix
    print("Extracting A from networkx graph")
    A = nx.adjacency_matrix(g_nx, nodelist=df.index, weight=None)
    # A = nx.adjacency_matrix(g_nx)
    print("...Done")
    return (X_all, y_all, A)


def load_polblogs(data_home="datasets/polblogs"):


    # load the graph
    g_nx = nx.read_gpickle(os.path.join(data_home, "polblogs_graph.gpickle"))

    # Load the labels
    y_all = np.load(os.path.join(data_home, "polblogs_labels.npy"))

    df_y = pd.DataFrame({"a": y_all, "b": 1 - y_all})
    y_all = df_y.values

    X_all = np.eye(g_nx.number_of_nodes())

    print("y_all shape {}".format(y_all.shape))
    print("X_all shape {}".format(X_all.shape))

    # Extract the adjacency matrix.
    # A should be scipy sparse matrix
    print("Extracting A from networkx graph")
    A = nx.to_scipy_sparse_matrix(g_nx, format="coo", dtype=np.int)
    # Remove self loops and set maximum value to 1
    adj = A.todense()
    adj[adj>1] = 1
    np.fill_diagonal(adj, 0)
    # convert back to scipy sparse matrix
    A = coo_matrix(adj)
    print("...Done")

    return (X_all, y_all, A)


def load_cora(data_home="datasets/feature_based_datasets/cora", cites_filename="cora.cites"):

    df = pd.read_csv(
        os.path.join(data_home, "cora.content"), sep=r"\s+", header=None, index_col=0
    )
    features_df = df.iloc[:, :-1]
    labels_df = df.iloc[:, -1]

    X_all = features_df.values

    y_all = LabelBinarizer().fit_transform(labels_df.values)

    fname_ext = cites_filename.split(".")[-1]

    if fname_ext == "adjlist":
        g_ = nx.read_adjlist(os.path.join(data_home, cites_filename))
        A = nx.to_scipy_sparse_matrix(g_, nodelist=sorted(df.index), format="coo")
    elif fname_ext == "gpickle":
        g_ = nx.read_gpickle(os.path.join(data_home, cites_filename))
        A = nx.to_scipy_sparse_matrix(g_, format="coo")
    else:
        edge_list_df = pd.read_csv(
            os.path.join(data_home, cites_filename), sep=r"\s+", header=None
        )

        idx_map = {j: i for i, j in enumerate(df.index)}

        H = nx.from_pandas_edgelist(edge_list_df, 0, 1)
        G = nx.relabel.relabel_nodes(H, idx_map)

        A = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G.nodes()), format="coo")

    return (X_all, y_all, A)


def load_citeseer(data_home="datasets/feature_based_datasets/citeseer"):

    df = pd.read_csv(
        os.path.join(data_home, "citeseer.content"),
        sep=r"\s+",
        header=None,
        index_col=0,
    )
    df.index = df.index.map(str)

    features_df = df.iloc[:, :-1]
    labels_df = df.iloc[:, -1]

    X_all = features_df.values

    y_all = LabelBinarizer().fit_transform(labels_df.values)

    edge_list_df = pd.read_csv(
        os.path.join(data_home, "citeseer.cites"), sep=r"\s+", dtype=str, header=None
    )

    idx_map = {j: i for i, j in enumerate(df.index)}

    H = nx.from_pandas_edgelist(edge_list_df, 0, 1)
    G = nx.relabel.relabel_nodes(H, idx_map)

    # This dataset has about 15 nodes in the edge list that don't have corresponding entries
    # in citeseer.content, that is don't have features. We need to identify them and then remove
    # them from the graph along with all the edges to/from them.
    nodes_to_remove = [n for n in G.nodes() if type(n) == str]
    G.remove_nodes_from(nodes_to_remove)

    A = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G.nodes()), format="coo")

    return (X_all, y_all, A)


def load_telco(data_home="datasets/telco"):

    df = pd.read_csv(
        os.path.join(data_home, "Telco-Customer-Churn-prepared.csv"), header=0
    )

    df.index = df["customerID"]
    df.drop(columns=["customerID"], inplace=True)

    y_all = df["Churn"].values
    y_all = LabelBinarizer().fit_transform(y_all)

    df.drop(columns=["Churn"], inplace=True)

    X_all = df.values

    n, d = X_all.shape
    # create the adjacency matrix
    A = coo_matrix((n, n))

    return (X_all, y_all, A)


def read_r_data(name, path):

    # importing here is naughty but need to isolate the side-effects
    # of importing, which in this case creates an interactive shell session...
    from rpy2.robjects import r

    load = r["load"]
    load(path)

    H = nx.from_numpy_array(np.asarray(r[name]))
    G = nx.relabel_nodes(H, dict(enumerate(r[name].colnames)))

    return G


def load_protein(data_home="datasets"):

    G = read_r_data(name="Y_Pro", path=os.path.join(data_home, "protein/Y_Pro.rda"))
    A = nx.to_scipy_sparse_matrix(G, format="coo")

    X_all = np.eye(G.number_of_nodes())
    y_all = np.zeros(shape=(G.number_of_nodes(), 2))

    return (X_all, y_all, A)


def load_genesis(data_home="datasets"):

    G = read_r_data(name="Y_Gen", path=os.path.join(data_home, "genesis/Y_Gen.rda"))

    A = nx.to_scipy_sparse_matrix(G, format="coo")

    X_all = np.eye(G.number_of_nodes())
    y_all = np.zeros(shape=(G.number_of_nodes(), 2))

    return (X_all, y_all, A)


def get_data(
    dataset_name,
    random_split,
    split_sizes,
    random_state,
    balanced_split=False,
    samples_per_class=20,
    edgelist=None,
    get_y_values=False
):

    tf.logging.info("Loading '{}' dataset...".format(dataset_name))
    loader = DATASET_LOADERS[dataset_name]
    X, y, A = loader()
    if dataset_name == "cora_edgelist":
        tf.logging.info("Loading with edgelist '{}'".format(edgelist))
        g_ = nx.read_gpickle(os.path.join("attacked_datasets/cora", edgelist))
        A = nx.adjacency_matrix(g_)
    elif dataset_name == "citeseer_edgelist":
        tf.logging.info("Loading with edgelist '{}'".format(edgelist))
        g_ = nx.read_gpickle(os.path.join("attacked_datasets/citeseer", edgelist))
        A = nx.adjacency_matrix(g_)
    elif dataset_name == "pubmed_edgelist":
        tf.logging.info("Loading with edgelist '{}'".format(edgelist))
        g_ = nx.read_gpickle(os.path.join("attacked_datasets/pubmed", edgelist))
        A = nx.adjacency_matrix(g_)
    elif dataset_name == "polblogs_edgelist":
        tf.logging.info("Loading with edgelist '{}'".format(edgelist))
        g_ = nx.read_gpickle(os.path.join("attacked_datasets/polblogs", edgelist))
        A = nx.adjacency_matrix(g_).astype(np.int)
        # Remove self loops and set maximum value to 1
        adj = A.todense()
        adj[adj > 1] = 1
        adj[adj < 0] = 0
        np.fill_diagonal(adj, 0)
        # convert back to scipy sparse matrix
        A = coo_matrix(adj)
    X = normalize(X, norm="l1", axis=1)

    n, d = X.shape
    _, k = y.shape

    tf.logging.info("Dataset has {} samples, dimensionality {}".format(n, d))
    tf.logging.info("Targets belong to {} classes".format(k))

    if random_split:
        split = recursive_stratified_shuffle_split(
            sizes=split_sizes, random_state=random_state
        )
        indices = list(split(X, y))
    elif balanced_split:
        indices = balanced_data_split(X, y, samples_per_class, random_state=random_state)
    else:
        if dataset_name in ("cora_knn", "cora_supervised", "cora_jaccard", "cora_tsne", "cora_cosine","cora_edgelist"):
            indices = load_split(name="cora")
        elif dataset_name in ('citeseer_edgelist'):
            indices = load_split("citeseer")
        elif dataset_name in ('pubmed_edgelist'):
            indices = load_split(name='pubmed')
        else:
            indices = load_split(dataset_name)

    tf.logging.info(
        "Split resulted in "
        "{} training, "
        "{} validation, "
        "{} test samples.".format(*map(len, indices))
    )

    [mask_train, mask_val, mask_test] = masks = list(
        map(partial(indices_to_mask, size=n), indices)
    )

    y_train, y_val, y_test = map(partial(mask_values, y), masks)

    A = A.toarray()
    if get_y_values:
        return X, y, A, mask_train, mask_val, mask_test, y_train, y_val, y_test, n, d, k

    return X, y, A, mask_train, mask_val, mask_test, n, d, k


def balanced_data_split(X, y, samples_per_class, random_state):
    indices = []

    num_classes = y.shape[1]
    train_indices = []

    for c in range(num_classes):
        ind = np.where(y[:, c] > 0.5)[0]
        #train_indices.extend(list(np.random.choice(ind, samples_per_class, replace=False)))
        train_indices.extend(list(random_state.choice(ind, samples_per_class, replace=False)))
        #print(ind)

    indices.append(train_indices)

    all_indices = set(range(y.shape[0]))
    # remove the indices of the data in the training set.
    unused_indices = all_indices - set(train_indices)

    # Now split the remaining data into validation and test sets using
    # uniform sampling such that validation data are 25% of the remaining data
    # and test data is 75% of the remaining data.
    val_size = int(len(unused_indices)*0.30)
    #val_indices = list(np.random.choice(list(unused_indices), val_size, replace=False))
    val_indices = list(random_state.choice(list(unused_indices), val_size, replace=False))

    test_indices = list(unused_indices-set(val_indices))

    # some sanity checks
    assert (len(train_indices)+len(val_indices)+len(test_indices)) == y.shape[0]
    assert len(set(train_indices).intersection(set(val_indices))) == 0
    assert len(set(val_indices).intersection(set(test_indices))) == 0

    indices.append(val_indices)
    indices.append(test_indices)


    return indices


DATASET_LOADERS = dict(
    karate_club=load_karate_club,
    iris=load_iris,
    cora_legacy=partial(load_cora, "datasets/feature_based_datasets/cora", "cora.cites"),
    cora_jaccard=partial(
        load_cora, "datasets/feature_based_datasets/cora", "cora_graph_jaccard.edgelist"
    ),
    cora_knn=partial(
        load_cora, "datasets/feature_based_datasets/cora", "cora_graph_supervised_knn.edgelist"
    ),
    cora_supervised=partial(
        load_cora, "datasets/feature_based_datasets/cora", "cora_graph_supervised.edgelist"
    ),
    cora_tsne=partial(load_cora, "datasets/feature_based_datasets/cora", "cora_graph_tsne.gpickle"),
    cora_cosine=partial(load_cora, "datasets/feature_based_datasets/cora", "cora_graph_cosine_knn_16.gpickle"),
    cora=partial(load_dataset, "cora"),
    cora_edgelist=partial(load_dataset,"cora"),
    citeseer_edgelist=partial(load_dataset,"citeseer"),
    citeseer=partial(load_dataset, "citeseer"),
    polblogs=partial(load_polblogs, "datasets/polblogs"),
    polblogs_edgelist=partial(load_polblogs, "datasets/polblogs"),
    pubmed=partial(load_dataset, "pubmed"),
    pubmed_edgelist=partial(load_dataset, "pubmed"),
    twitter=load_twitter,
    telco=load_telco,
    protein=load_protein,
    genesis=load_genesis,
)
