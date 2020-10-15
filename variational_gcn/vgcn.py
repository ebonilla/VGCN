import warnings
import os.path

import tensorflow as tf
import tensorflow.keras.backend as K


from variational_gcn.graph_priors import get_prior

from variational_gcn.graph_posteriors import (
    get_variational_posterior,
    sample_posterior,
)

from variational_gcn.likelihood import (
    get_conditional_likelihood,
    get_conditional_likelihood_kronecker,
    predict,
    get_predictive_distribution
)

from variational_gcn.losses import get_losses

from variational_gcn.metrics import evaluate_accuracy, evaluate_mnlp

from variational_gcn.utils.io import (
    save_posterior,
    get_results_handler,
    get_intermediate_results,
    get_final_results,
)


class VariationalGCN:

    def __init__(self, X, y, A, mask_train, mask_val, mask_test, n, d, k,
                 degree, dropout_rate, layer_type, l2_reg_scale_gcn,
                 prior_type, constant, one_smoothing_factor, zero_smoothing_factor,
                 knn_k, knn_metric, relaxed, temperature_prior, init_size, init_val,
                 posterior_type, temperature_posterior, latent_dim, logit_shift, beta,
                 mc_samples_train, mc_samples_test):
        """
        Variational inference for a GCN Model (semi-supervised classification setting)
        :param X: (n,d) float array of features
        :param y: (n,k) float array of one-hot-encoded labels
        :param A: (n, n) float given adjacency matrix
        :param mask_train: (bool) array training mask
        :param mask_val: (bool) array validation mask
        :param mask_test: (bool) array test mask
        :param n: (int) number of instances/onodes
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
        :param init_size: (int) initial size of free_lowdim posterior (not used in ICML submission)
        :param init_val: (float) initial size value of free_lowdim posterior (not used in ICML submission)
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

        self.x = tf.placeholder(dtype=tf.float32, shape=(n, d))

        prior, self.probs_tril, probs = get_prior(prior_type, X, A, constant, n,
                                                  one_smoothing_factor, zero_smoothing_factor,
                                                  knn_k, knn_metric, relaxed, temperature_prior, init_size=init_size,
                                                  init_val=init_val)

        posterior, self.posterior_param = get_variational_posterior(
            relaxed,
            posterior_type,
            temperature_posterior,
            self.x,
            probs,
            self.probs_tril,
            n,
            latent_dim,
            logit_shift,
            use_graph,
        )

        a_sample, a_sample_tril, b_sample_tril = sample_posterior(posterior, relaxed, mc_samples_train)

        # apply Kronecker operations in the likelihood
        if prior_type == "free_lowdim" and posterior_type == "free_lowdim":
            likelihood, self.gcn = get_conditional_likelihood_kronecker(
                self.x, a_sample, k, degree, dropout_rate, layer_type, l2_reg_scale_gcn, n, init_size=init_size
            )
        else:  # standard gcn
            likelihood, self.gcn = get_conditional_likelihood(
                self.x, a_sample, k, degree, dropout_rate, layer_type, l2_reg_scale_gcn
            )

        self.elbo_train, self.elbo_val, self.elbo_test, self.loss_train, self.loss_val, self.loss_test, \
            self.reg, self.kl, self.ell_train = get_losses(prior,
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
                                                           beta
                                                           )

        if mc_samples_train == mc_samples_test:  # reuse samples from training
            self.y_pred = predict(likelihood)
        else:  # build new graph using fresh samples
            self.y_pred = get_predictive_distribution(self.x, self.gcn, posterior, relaxed, mc_samples_test)

        self.accuracy_train, self.accuracy_val, self.accuracy_test = evaluate_accuracy(
            y, self.y_pred, mask_train, mask_val, mask_test
        )

        self.mnlp_train, self.mnlp_val, self.mnlp_test = evaluate_mnlp(
            y, self.y_pred, mask_train, mask_val, mask_test
        )

    def train_and_predict(self, X, y, num_epochs, experimental,
                          initial_learning_rate,
                          optimizer_name,
                          log_every_n_iter,
                          results_dir,
                          checkpoint_dir,
                          experiment_id,
                          summary_dir,
                          save_checkpoint_steps,
                          save_summaries_steps,
                          alternate_optimization,
                          gcn_opt_steps,
                          adj_opt_steps):
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
        :param alternate_optimization: (bool) whether to carry out alternate optimization of ['gcn', 'adj'] params
        :param gcn_opt_steps: (int) number of steps to optimize GCN parameters (if alternate_optimization=True)
        :param adj_opt_steps: (int) number of steps to optimize posterior-adj params (if alternate_optimization=True)
        :return:
        """

        tf.summary.scalar("accuracy/train", self.accuracy_train)
        tf.summary.scalar("accuracy/val", self.accuracy_val)
        tf.summary.scalar("accuracy/test", self.accuracy_test)
        tf.summary.scalar("mnlp/train", self.mnlp_train)
        tf.summary.scalar("mnlp/val", self.mnlp_val)
        tf.summary.scalar("mnlp/test", self.mnlp_test)

        tf.summary.scalar("loss/elbo/train", tf.reduce_sum(self.elbo_train))
        tf.summary.scalar("loss/elbo/val", tf.reduce_sum(self.elbo_val))
        tf.summary.scalar("loss/elbo/test", tf.reduce_sum(self.elbo_test))

        tf.summary.scalar("loss/train", self.loss_train)
        tf.summary.scalar("loss/val", self.loss_val)
        tf.summary.scalar("loss/test", self.loss_test)

        tf.summary.scalar("loss/kl_train", self.kl)
        tf.summary.scalar("loss/ell_train", self.ell_train)
        tf.summary.scalar("loss/reg_train", self.reg)

        global_step = tf.train.get_or_create_global_step()

        if experimental:

            train_op = tf.contrib.layers.optimize_loss(
                self.loss_train,
                global_step=global_step,
                learning_rate=initial_learning_rate,
                optimizer=optimizer_name,
                summaries=["gradients"],
            )

            train_op_gcn = tf.contrib.layers.optimize_loss(self.loss_train,
                                                           global_step=global_step,
                                                           learning_rate=initial_learning_rate,
                                                           optimizer=optimizer_name,
                                                           summaries=["gradients"],
                                                           variables=self.gcn.trainable_weights
                                                           )

            train_op_adj = tf.contrib.layers.optimize_loss(self.loss_train,
                                                           global_step=global_step,
                                                           learning_rate=initial_learning_rate,
                                                           optimizer=optimizer_name,
                                                           summaries=["gradients"],
                                                           variables=self.posterior_param
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
            train_op = optimizer.minimize(self.loss_train, global_step=global_step)

            train_op_gcn = optimizer.minimize(self.loss_train, global_step=global_step,
                                              var_list=self.gcn.trainable_weights)
            train_op_adj = optimizer.minimize(self.loss_train, global_step=global_step,
                                              var_list=self.posterior_param)

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
                loss_train=self.loss_train,
                loss_val=self.loss_val,
                loss_test=self.loss_test,
                accuracy_train=self.accuracy_train,
                accuracy_val=self.accuracy_val,
                accuracy_test=self.accuracy_test,
                kl_train=self.kl,
                ell_train=self.ell_train,
                reg_train=self.reg,
                mnlp_train=self.mnlp_train,
                mnlp_val=self.mnlp_val,
                mnlp_test=self.mnlp_test,
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

        metrics_list = [self.loss_train, self.loss_val, self.loss_test,
                        self.accuracy_train, self.accuracy_val, self.accuracy_test,
                        self.kl, self.ell_train, self.reg,
                        self.mnlp_train, self.mnlp_val, self.mnlp_test]

        final_metrics_list = [self.loss_train, self.loss_val, self.loss_test,
                              self.accuracy_train, self.accuracy_val, self.accuracy_test,
                              self.mnlp_train, self.mnlp_val, self.mnlp_test]

        with tf.train.MonitoredTrainingSession(
                hooks=[logger],
                checkpoint_dir=checkpoint_dir,
                summary_dir=checkpoint_dir if summary_dir is None else summary_dir,
                save_checkpoint_steps=save_checkpoint_steps,
                save_summaries_steps=save_summaries_steps,
        ) as sess:
            if alternate_optimization is True:
                epoch = 0
                while epoch < num_epochs:
                    gcn_step = 0
                    adj_step = 0
                    while gcn_step < gcn_opt_steps and epoch < num_epochs:
                        sess.run(train_op_gcn, feed_dict={self.x: X, K.learning_phase(): True})
                        get_intermediate_results(sess, metrics_list, self.x, X, epoch, results_filename)
                        gcn_step += 1
                        epoch += 1
                    while adj_step < adj_opt_steps and epoch < num_epochs:
                        get_intermediate_results(sess, metrics_list, self.x, X, epoch, results_filename)
                        sess.run(train_op_adj, feed_dict={self.x: X, K.learning_phase(): True})
                        adj_step += 1
                        epoch += 1
            else:
                if checkpoint_dir is not None:  # saves initial posterior
                    save_posterior(sess, self.probs_tril, self.posterior_param, self.x, X, checkpoint_dir)
                for epoch in range(num_epochs):
                    sess.run(train_op, feed_dict={self.x: X, K.learning_phase(): True})
                    get_intermediate_results(sess, metrics_list, self.x, X, epoch, results_filename)

            get_final_results(sess, no_op, self.x, X, final_metrics_list, self.y_pred, y, self.mask_test, results_dir)
