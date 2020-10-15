import warnings
import os.path

import random
import tensorflow as tf
import tensorflow.keras.backend as K


from variational_gcn.graph_priors import get_prior_cluster

from variational_gcn.graph_posteriors import get_variational_posterior_cluster, sample_posterior

from variational_gcn.likelihood import (
    get_conditional_likelihood_cluster,
    predict_cluster,
    get_predictive_distribution_cluster,
)

from variational_gcn.losses import get_losses_cluster

from variational_gcn.metrics import evaluate_accuracy_cluster, evaluate_mnlp_cluster

from variational_gcn.utils.io import (
    save_posterior,
    get_results_handler,
    get_intermediate_results_cluster,
    get_final_results_cluster,
)


class ClusterVariationalGCN:
    def __init__(
        self,
        X,
        y,
        A,
        mask_train,
        mask_val,
        mask_test,
        n,
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
    ):
        """
        Variational inference for a GCN Model (semi-supervised classification setting)
        :param X: (n,d) list of float array of features
        :param y: (n,k) list of float array of one-hot-encoded labels
        :param A: (n, n) list of float given adjacency matrix
        :param mask_train: (bool) list of array training mask
        :param mask_val: (bool) list of array validation mask
        :param mask_test: (bool) list of array test mask
        :param n: (int) list of number of instances/onodes
        :param d: (int) feature dimensionality
        :param k: (int) number of classes
        :param degree: (int) Chebyshev polynomial expansion degree (None for standard GCN)
        :param dropout_rate: (float) dropout rate
        :param layer_type: (string) layer type (e.g. dense)
        :param l2_reg_scale_gcn: (float) L2 regularizarion for GCN weights
        :param prior_type: prior (string) type ["smoothing", "feature", "knn", "free_lowdim"]
        :param constant: (float) if not None, fill all off-diagonal entries of adjacency matrix
        :param one_smoothing_factor: (float) smoothing factor for ones in adjacency matrix prior
        :param zero_smoothing_factor: (float) smoothing factor for zeros in adjacency matrix prior
        :param knn_k: (int) number of neighbours for KNNG prior
        :param knn_metric: (string) distance metric to be used in KNNG prior ["cosine", "minkowski"]
        :param relaxed: (bool) whether to used  relaxed binary Concrete (True) or discrete distributions
        :param temperature_prior: (float)  temperature for prior binary Concrete distribution
        :param posterior_type: (string) posterior type ["free", "amortized", "lowrank", "kronecker", "free_lowdim"]
        :param temperature_posterior: (float)  temperature for posterior binary Concrete distribution
        :param latent_dim: (int) number of dimensions for low-rank posterior
        :param logit_shift: (float) offset parameter for low-rank posterior
        :param beta: scale (float) parameter for KL term, usually 0 < beta < 1
        :param mc_samples_train: (int) number of MC samples for estimating expectations in training
        :param mc_samples_test: (int) number of samples for estimating expectations in prediction
        """

        if constant is None:
            use_graph = True
        else:
            use_graph = False

        self.mask_test = mask_test

        self.xs = []
        for _, n_ in zip(A, n):
            self.xs.append(tf.placeholder(dtype=tf.float32, shape=(n_, d)))

        prior, self.probs_tril, probs = get_prior_cluster(
            prior_type,
            X,
            A,
            constant,
            n,
            one_smoothing_factor,
            zero_smoothing_factor,
            knn_k,
            knn_metric,
            relaxed,
            temperature_prior,
        )

        posterior, self.posterior_param = get_variational_posterior_cluster(
            relaxed,
            posterior_type,
            temperature_posterior,
            self.xs,
            probs,
            self.probs_tril,
            n,
            latent_dim,
            logit_shift,
            use_graph,
        )

        a_sample = []
        a_sample_tril = []
        b_sample_tril = []

        for posterior_ in posterior:
            a_sample_, a_sample_tril_, b_sample_tril_ = sample_posterior(
                posterior_, relaxed, mc_samples_train
            )
            a_sample.append(a_sample_)
            a_sample_tril.append(a_sample_tril_)
            b_sample_tril.append(b_sample_tril_)

        likelihood, self.gcn = get_conditional_likelihood_cluster(
            self.xs, a_sample, k, degree, dropout_rate, layer_type, l2_reg_scale_gcn
        )

        self.elbo_train, self.elbo_val, self.elbo_test, self.loss_train, self.loss_val, self.loss_test, self.reg, self.kl, self.ell_train = get_losses_cluster(
            prior,
            posterior,
            likelihood,
            y,
            a_sample_tril,
            b_sample_tril,
            self.gcn,
            mask_train,
            mask_val,
            mask_test,
            relaxed,
            beta=beta,
        )

        if mc_samples_train == mc_samples_test:  # reuse samples from training
            self.y_pred = predict_cluster(likelihood)
        else:  # build new graph using fresh samples
            self.y_pred = get_predictive_distribution_cluster(
                self.x, self.gcn, posterior, relaxed, mc_samples_test
            )

        self.accuracy_train, self.accuracy_val, self.accuracy_test = evaluate_accuracy_cluster(
            y, self.y_pred, mask_train, mask_val, mask_test
        )

        self.mnlp_train, self.mnlp_val, self.mnlp_test = evaluate_mnlp_cluster(
            y, self.y_pred, mask_train, mask_val, mask_test
        )

    def train_and_predict(
        self,
        X,
        y,
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
    ):
        """
        Trains a VGCN model and makes predictions
        :param X: (n,d) float array of features
        :param y: (n,k) float array of one-hot-encoded labels
        :param num_epochs: (int) number of training epoch
        :param experimental: (bool) wether to eun on experimental model
        :param initial_learning_rate: (float) initial learning rate for optimizer
        :param optimizer_name: (string) optimizer name (e.g. Adam). It can be different to Adam if experimental=True
        :param log_every_n_iter: (int) frequency (in epochs) to log results
        :param results_dir: (string) target directory where to save the results
        :param checkpoint_dir: (string) target directory where to save mode check points
        :param experiment_id: (string) experiment ID
        :param summary_dir: (string) target directory where to save summaries
        :param save_checkpoint_steps: (string) frequency (in epochs) to save check point
        :param save_summaries_steps: (int) frequency (in epochs) to save summaries
        :return:
        """

        tf.summary.scalar("accuracy/train", tf.add_n(self.accuracy_train))
        tf.summary.scalar("accuracy/val", tf.add_n(self.accuracy_val))
        tf.summary.scalar("accuracy/test", tf.add_n(self.accuracy_test))
        tf.summary.scalar("mnlp/train", tf.add_n(self.mnlp_train))
        tf.summary.scalar("mnlp/val", tf.add_n(self.mnlp_val))
        tf.summary.scalar("mnlp/test", tf.add_n(self.mnlp_test))

        # TODO: This one is tricky because elbo_* is list. What is the dimensionality of elbo_*? Can I call tf.add_n first?
        tf.summary.scalar("loss/elbo/train", tf.reduce_sum(self.elbo_train))
        tf.summary.scalar("loss/elbo/val", tf.reduce_sum(self.elbo_val))
        tf.summary.scalar("loss/elbo/test", tf.reduce_sum(self.elbo_test))

        tf.summary.scalar("loss/train", tf.add_n(self.loss_train))
        tf.summary.scalar("loss/val", tf.add_n(self.loss_val))
        tf.summary.scalar("loss/test", tf.add_n(self.loss_test))

        tf.summary.scalar("loss/kl_train", tf.add_n(self.kl))
        tf.summary.scalar("loss/ell_train", tf.add_n(self.ell_train))
        tf.summary.scalar("loss/reg_train", tf.add_n(self.reg))

        global_step = tf.train.get_or_create_global_step()

        if experimental:

            train_op = []
            for loss_train_ in self.loss_train:
                train_op.append(
                    tf.contrib.layers.optimize_loss(
                        loss_train_,
                        global_step=global_step,
                        learning_rate=initial_learning_rate,
                        optimizer=optimizer_name,
                        summaries=["gradients"],
                    )
                )
        else:

            if optimizer_name != "Adam":
                warnings.warn(
                    (
                        "Optimizer '{}' only available in experimental mode. "
                        "Defaulting to 'Adam'."
                    ).format(optimizer_name)
                )

            optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate)
            train_op = []
            for loss_train_ in self.loss_train:
                train_op.append(
                    optimizer.minimize(loss_train_, global_step=global_step)
                )

        if checkpoint_dir is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, experiment_id)

        if results_dir is None:
            results_dir = checkpoint_dir
            results_dir = os.path.join(results_dir, experiment_id)

        header = (
            "time, epoch, loss_train, loss_val, loss_test, accuracy_train, accuracy_val, accuracy_test, "
            + "kl_train, ell_train, reg_train, mnlp_train, mnlp_val, mnlp_test"
        )

        results_filename = get_results_handler(results_dir, header)

        # global_step == epoch since each step is full pass over all data
        logger = tf.train.LoggingTensorHook(
            dict(
                epoch=global_step,
                loss_train=tf.add_n(self.loss_train),
                loss_val=tf.add_n(self.loss_val),
                loss_test=tf.add_n(self.loss_test),
                accuracy_train=tf.add_n(self.accuracy_train),
                accuracy_val=tf.add_n(self.accuracy_val),
                accuracy_test=tf.add_n(self.accuracy_test),
                kl_train=tf.add_n(self.kl),
                ell_train=tf.add_n(self.ell_train),
                reg_train=tf.add_n(self.reg),
                mnlp_train=tf.add_n(self.mnlp_train),
                mnlp_val=tf.add_n(self.mnlp_val),
                mnlp_test=tf.add_n(self.mnlp_test),
                learning_phase=K.learning_phase(),
            ),
            every_n_iter=log_every_n_iter,
            formatter=lambda tensors: (
                "epoch={epoch:04d}, "
                "loss={loss_train:04f}, "
                "loss_val={loss_val:04f}, "
                "loss_test={loss_test:04f}, "
                "acc={accuracy_train:04f}, "
                "acc_val={accuracy_val:04f}, "
                "acc_test={accuracy_test:04f}, "
                "kl_train={kl_train:04f}, "
                "ell_train={ell_train:04f}, "
                "reg_train={reg_train:04f}, "
                "mnlp_train={mnlp_train:04f}, "
                "mnlp_val={mnlp_val:04f}, "
                "mnlp_test={mnlp_test:04f}, "
                "learning_phase={learning_phase}"
            ).format(**tensors),
        )

        no_op = tf.no_op()

        metrics_list = []

        for (
            loss_train_,
            loss_val_,
            loss_test_,
            accuracy_train_,
            accuracy_val_,
            accuracy_test_,
            kl_,
            ell_train_,
            reg_,
            mnlp_train_,
            mnlp_val_,
            mnlp_test_,
        ) in zip(
            self.loss_train,
            self.loss_val,
            self.loss_test,
            self.accuracy_train,
            self.accuracy_val,
            self.accuracy_test,
            self.kl,
            self.ell_train,
            self.reg,
            self.mnlp_train,
            self.mnlp_val,
            self.mnlp_test,
        ):
            metrics_list.append(
                [
                    loss_train_,
                    loss_val_,
                    loss_test_,
                    accuracy_train_,
                    accuracy_val_,
                    accuracy_test_,
                    kl_,
                    ell_train_,
                    reg_,
                    mnlp_train_,
                    mnlp_val_,
                    mnlp_test_,
                ]
            )

        final_metrics_list = []
        for (
            loss_train_,
            loss_val_,
            loss_test_,
            accuracy_train_,
            accuracy_val_,
            accuracy_test_,
            mnlp_train_,
            mnlp_val_,
            mnlp_test_,
        ) in zip(
            self.loss_train,
            self.loss_val,
            self.loss_test,
            self.accuracy_train,
            self.accuracy_val,
            self.accuracy_test,
            self.mnlp_train,
            self.mnlp_val,
            self.mnlp_test,
        ):
            final_metrics_list.append(
                [
                    loss_train_,
                    loss_val_,
                    loss_test_,
                    accuracy_train_,
                    accuracy_val_,
                    accuracy_test_,
                    mnlp_train_,
                    mnlp_val_,
                    mnlp_test_,
                ]
            )

        # TODO: I cannot use the logger because it needs to know all the tensor values for all subgraph every time sess.run
        #  is called.
        with tf.train.MonitoredTrainingSession(
                # hooks=[logger],
                checkpoint_dir=checkpoint_dir,
                summary_dir=checkpoint_dir if summary_dir is None else summary_dir,
                save_checkpoint_steps=save_checkpoint_steps,
                save_summaries_steps=save_summaries_steps,
        ) as sess:
            if checkpoint_dir is not None:
                save_posterior(sess, self.posterior_param, self.xs, X, checkpoint_dir)
            for epoch in range(num_epochs):
                input_data = list(zip(train_op, self.xs, X))
                random.shuffle(input_data)

                for train_op_, x, features in input_data:
                    sess.run(train_op_, feed_dict={x: features, K.learning_phase(): True})

                get_intermediate_results_cluster(
                    sess, metrics_list, self.xs, X, epoch, results_filename
                )

            get_final_results_cluster(
                sess, no_op, self.xs, X, final_metrics_list, self.y_pred, y, self.mask_test, results_dir
            )
