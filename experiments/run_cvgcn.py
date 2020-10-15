# -*- coding: utf-8 -*-

"""Console script for cluster variational_gcn."""
import click
import random
import metis
import numpy as np
import tensorflow as tf

from variational_gcn.layers import LAYER_CLS_NAMES
from variational_gcn.datasets import DATASET_LOADERS, get_data
import variational_gcn.config as cfg

from variational_gcn.utils.base import (
    corrupt_adjacency,
    get_experiment_ID,
    split_train_val,
)
from variational_gcn.utils.io import save_parameters, print_settings
from variational_gcn.cvgcn import ClusterVariationalGCN

tf.logging.set_verbosity(tf.logging.INFO)


def get_random_clusters(A, num_clusters=2):
    """
        It splits a graph represented by the adjacency matrix A into num_clusters number of subgraphs.
        :param A: The original graph adjacency matrix
        :param num_clusters: The number of subgraphs
        :return: <list> Returns a list of num_clusters lists such that the i-th list holds the row indices of the nodes
            that belong to the i-th subgraph.
        """
    all_nodes = list(range(A.shape[0]))
    random.shuffle(all_nodes)
    cluster_size = len(all_nodes) // num_clusters
    clusters = [
        all_nodes[i : i + cluster_size] for i in range(0, len(all_nodes), cluster_size)
    ]

    if len(clusters) > num_clusters:
        # for the case that the number of nodes is not exactly divisible by num_clusters, we combine
        # the last cluster with the second last one
        clusters[-2].extend(clusters[-1])
        del clusters[-1]

    print(f"Number of clusters {num_clusters}")
    for i, c in enumerate(clusters):
        print(f"{i} cluster has size {len(c)}")

    return clusters


def get_subgraph_data(X, y, A, mask_train, mask_val, mask_test, cluster_indices):
    As = []
    Xs = []
    ys = []
    ns = []
    masks_train = []
    masks_val = []
    masks_test = []
    for subgraph in cluster_indices:
        subgraph_ar = np.array(subgraph)
        As.append(A[subgraph_ar[:, None], subgraph])
        Xs.append(X[subgraph_ar, :])
        ys.append(y[subgraph_ar, :])
        masks_train.append(mask_train[subgraph_ar])
        masks_val.append(mask_val[subgraph_ar])
        masks_test.append(mask_test[subgraph_ar])
        ns.append(len(As[-1]))

    d = Xs[0].shape[1]

    return As, Xs, ys, masks_train, masks_test, masks_val, ns, d


@click.command()
@click.argument("name")
@click.option(
    "--dataset-name",
    default=cfg.DATASET_NAME,
    type=click.Choice(DATASET_LOADERS.keys()),
)
@click.option(
    "--random-split/--load-split",
    default=cfg.RANDOM_SPLIT,
    help="Use stratified randomly-shuffled split or load split.",
)
@click.option(
    "--balanced-split/--no-balanced-split",
    default=cfg.BALANCED_SPLIT,
    help="Use balanced training set.",
)
@click.option(
    "--samples-per-class",
    default=cfg.SAMPLES_PER_CLASS,
    type=click.INT,
    help="Number of training samples per class for option --balanced-split.",
)
@click.option(
    "--adj-corruption-method",
    default=cfg.ADJ_CORRUPTION_METHOD,
    type=click.Choice(cfg.adj_corruption_methods),
)
@click.option("--perc-corruption", default=cfg.PERC_CORRUPTION, type=click.FLOAT)
# TODO: should support both FLOAT and INT
@click.option("--split-sizes", default=cfg.SPLIT_SIZES, nargs=2, type=click.FLOAT)
@click.option("--train-range", nargs=2, default=cfg.TRAIN_RANGE, type=click.INT)
@click.option(
    "--validation-range", nargs=2, default=cfg.VALIDATION_RANGE, type=click.INT
)
@click.option("--test-range", nargs=2, default=cfg.TEST_RANGE, type=click.INT)
@click.option(
    "--optimizer-name",
    default=cfg.OPTIMIZER_NAME,
    type=click.Choice(tf.contrib.layers.OPTIMIZER_CLS_NAMES.keys()),
)
@click.option(
    "--temperature-prior",
    default=cfg.TEMPERATURE_PRIOR,
    type=click.FLOAT,
    help="Relaxed Bernoulli prior temperature parameter.",
)
@click.option(
    "--temperature-posterior",
    default=cfg.TEMPERATURE_POSTERIOR,
    type=click.FLOAT,
    help="Relaxed Bernoulli variational posterior temperature parameter.",
)
@click.option(
    "--mc-samples-train",
    default=cfg.MC_SAMPLES_TRAIN,
    type=click.INT,
    help="Number of Monte Carlo samples.",
)
@click.option(
    "--mc-samples-test",
    default=cfg.MC_SAMPLES_TEST,
    type=click.INT,
    help="Number of Monte Carlo samples.",
)
@click.option(
    "--initial-learning-rate",
    default=cfg.INITIAL_LEARNING_RATE,
    type=click.FLOAT,
    help="Initial learning rate.",
)
@click.option(
    "--layer-type",
    default=cfg.LAYER_CLS_NAME,
    type=click.Choice(LAYER_CLS_NAMES.keys()),
)
@click.option(
    "--degree",
    default=cfg.DEGREE,
    type=click.INT,
    help="Degree of Chebyshev polynomial expansion.",
)
@click.option(
    "--dropout-rate",
    default=cfg.DROPOUT_RATE,
    type=click.FLOAT,
    help="Dropout rate. Fraction of the input units to drop.",
)
@click.option(
    "--l2-reg-scale-gcn",
    default=cfg.L2_REG_SCALE_GCN,
    type=click.FLOAT,
    help="L2 regularizer on GCN Parameters",
)
@click.option(
    "--constant",
    default=cfg.CONSTANT,
    type=click.FLOAT,
    help="Fix all off-diagonal entries of adjacency matrix to constant.",
)
@click.option(
    "--prior-type", default=cfg.DEFAULT_PRIOR_TYPE, type=click.Choice(cfg.PRIOR_TYPES)
)
@click.option(
    "--one-smoothing-factor",
    default=cfg.ONE_SMOOTHING_FACTOR,
    type=click.FLOAT,
    help="Factor for smoothing ones in binary adjacency matrix to float.",
)
@click.option(
    "--zero-smoothing-factor",
    default=cfg.ZERO_SMOOTHING_FACTOR,
    type=click.FLOAT,
    help="Factor for smoothing zeros in binary adjacency matrix to float.",
)
@click.option(
    "--knn-metric",
    default=cfg.DEFAULT_KNN_METRIC,
    type=click.Choice(cfg.KNN_METRICS),
    help="Default knn metric to use for building prior",
)
@click.option(
    "--knn-k",
    default=cfg.DEFAULT_KNN_K,
    type=click.INT,
    help="Default number of neighbours for KNN prior.",
)
@click.option(
    "--num-epochs",
    default=cfg.NUM_EPOCHS,
    type=click.INT,
    help="Number of training epochs.",
)
@click.option(
    "--posterior-type",
    default=cfg.DEFAULT_POSTERIOR_TYPE,
    type=click.Choice(cfg.POSTERIOR_TYPES),
)
@click.option("--latent-dim", default=cfg.LATENT_DIM, type=click.INT)
@click.option("--logit-shift", default=cfg.LOGIT_SHIFT, type=click.FLOAT)
@click.option(
    "--relaxed/--discrete",
    default=cfg.RELAXED,
    help="Use continuous relaxed distributions with reparameterization gradients, otherwise use discrete distributions.",
)
@click.option(
    "--log-every-n-iter",
    default=cfg.LOG_EVERY_N_ITER,
    type=click.INT,
    help="Log every n iterations (epochs).",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=cfg.CHECKPOINT_DIR,
    help="Model checkpoint directory.",
)
@click.option(
    "--summary-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Summary directory.",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=cfg.RESULTS_DIR,
    help="Results directory.",
)
@click.option(
    "--save-checkpoint-steps",
    default=cfg.SAVE_CHECKPOINT_STEPS,
    type=click.INT,
    help="The frequency, in steps, that a checkpoint is saved.",
)
@click.option(
    "--save-summaries-steps",
    default=cfg.SAVE_SUMMARIES_STEPS,
    type=click.INT,
    help="The frequency, in steps, that summaries are saved.",
)
@click.option(
    "--experimental/--no-experimental",
    default=cfg.EXPERIMENTAL,
    help="Use experimental TensorFlow functionality.",
)
@click.option(
    "--reg-scale-l1",
    default=cfg.REGULARIZER_L1_SCALE,
    type=click.FLOAT,
    help="Scale parameter for variational parameters l1 regularizer. The default is not to use l1 regularisation.",
)
@click.option(
    "--reg-scale-l2",
    default=cfg.REGULARIZER_L2_SCALE,
    type=click.FLOAT,
    help="Scale parameter for variational parameters l2 regularizer. The default is not to use l2 regularisation.",
)
@click.option(
    "--load-adj-matrix",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False),
    help="An adjaceny matrix to load",
)
@click.option(
    "--edgelist",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False),
    help="A gpickled adjacency matrix to load",
)
@click.option(
    "--use-half-val-to-train/--no-use-half-val-to-train",
    default=cfg.USE_HALF_VAL_TO_TRAIN,
    help="Use half the validation data in training.",
)
@click.option("-s", "--seed", default=cfg.SEED, type=click.INT, help="Random seed")
@click.option(
    "-s", "--seed-np", default=cfg.SEED_NP, type=click.INT, help="Random seed for numpy"
)
@click.option(
    "-s",
    "--seed-val",
    default=cfg.SEED_VAL,
    type=click.INT,
    help="Random seed for splitting the validation set",
)
@click.option(
    "--beta",
    default=cfg.KL_FACTOR,
    type=click.FLOAT,
    help="Scale parameter for KL term.",
)
@click.option(
    "--metis-clustering/--random-clustering",
    default=cfg.USE_METIS,
    help="Use METIS or Random graph clustering.",
)
@click.option(
    "--num-clusters",
    default=cfg.NUM_CLUSTERS,
    type=click.INT,
    help="Number of subgraphs/clusters.",
)
def main(
    name,
    dataset_name,
    random_split,
    balanced_split,
    samples_per_class,
    adj_corruption_method,
    perc_corruption,
    split_sizes,
    train_range,
    validation_range,
    test_range,
    optimizer_name,
    temperature_prior,
    temperature_posterior,
    mc_samples_train,
    mc_samples_test,
    initial_learning_rate,
    layer_type,
    degree,
    dropout_rate,
    l2_reg_scale_gcn,
    constant,
    prior_type,
    one_smoothing_factor,
    zero_smoothing_factor,
    knn_metric,
    knn_k,
    num_epochs,
    posterior_type,
    latent_dim,
    logit_shift,
    relaxed,
    log_every_n_iter,
    checkpoint_dir,
    summary_dir,
    results_dir,
    save_checkpoint_steps,
    save_summaries_steps,
    experimental,
    reg_scale_l1,
    reg_scale_l2,
    load_adj_matrix,
    edgelist,
    use_half_val_to_train,
    seed,
    seed_np,
    seed_val,
    beta,
    metis_clustering,
    num_clusters,
):
    tf.random.set_random_seed(seed)
    random_state = np.random.RandomState(seed_np)

    experiment_id = get_experiment_ID()
    params = print_settings(experiment_id)
    save_parameters(results_dir, params)

    X, y, A, mask_train, mask_val, mask_test, n, d, k = get_data(
        dataset_name=dataset_name,
        random_split=random_split,
        split_sizes=split_sizes,
        balanced_split=balanced_split,
        samples_per_class=samples_per_class,
        random_state=random_state,
        edgelist=edgelist,
    )

    if use_half_val_to_train:
        mask_train, mask_val = split_train_val(mask_train, mask_val, seed_val)

    A = corrupt_adjacency(A, adj_corruption_method, perc_corruption, load_adj_matrix)

    if metis_clustering and num_clusters > 1:
        node_ids = np.arange(len(A))
        adjlist = [tuple(np.where(neighbours)[0]) for neighbours in A]

        edgecuts, parts = metis.part_graph(adjlist, num_clusters)
        parts = np.array(parts)
        cluster_indices = []
        cluster_ids = np.unique(parts)
        for cluster_id in cluster_ids:
            mask = np.where(parts == cluster_id)
            cluster_indices.append(node_ids[mask])
    else:
        cluster_indices = get_random_clusters(A, num_clusters=num_clusters)

    As, Xs, ys, masks_train, masks_test, masks_val, ns, d = get_subgraph_data(
        X, y, A, mask_train, mask_val, mask_test, cluster_indices
    )

    vgcn = ClusterVariationalGCN(
        Xs,
        ys,
        As,
        masks_train,
        masks_val,
        masks_test,
        ns,
        d,
        k,
        degree,
        dropout_rate,
        layer_type,
        l2_reg_scale_gcn,
        prior_type,
        constant,
        one_smoothing_factor,
        zero_smoothing_factor,
        knn_k,
        knn_metric,
        relaxed,
        temperature_prior,
        posterior_type,
        temperature_posterior,
        latent_dim,
        logit_shift,
        beta,
        mc_samples_train,
        mc_samples_test,
    )

    vgcn.train_and_predict(
        Xs,
        ys,
        num_epochs,
        experimental,
        initial_learning_rate,
        optimizer_name,
        log_every_n_iter,
        results_dir,
        checkpoint_dir,
        experiment_id,
        summary_dir,
        save_checkpoint_steps,
        save_summaries_steps,
    )

    return 0


if __name__ == "__main__":
    exit(main())  # pragma: no cover
