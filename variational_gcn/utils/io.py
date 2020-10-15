import pickle
import yaml
import click

import os
import csv

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import tensorflow.keras.backend as K

import datetime


def print_settings(experiment_id):
    params = click.get_current_context().params
    params["experiment_id"] = experiment_id
    click.secho(
        "Summary of all parameter settings:\n"
        "----------------------------------\n"
        "{}".format(yaml.dump(params, default_flow_style=False)),
        fg="yellow",
    )
    return params


def get_final_results(
    sess, no_op, x, X, final_metrics_list, y_pred, y, mask_test, results_dir
):
    # Calling no-op to trigger logging tensor hook to print metrics
    # in prediction phase.
    results = sess.run(final_metrics_list, feed_dict={x: X, K.learning_phase(): False})
    sess.run(no_op, feed_dict={x: X})  # triggering Keras prediction phase (no Dropout)
    predictions = sess.run([y_pred], feed_dict={x: X})
    save_final_results(results_dir, results, predictions, y, mask_test)


def get_intermediate_results(sess, metrics_list, x, X, epoch, results_filename):
    if (
        results_filename is not None
    ):  # predictions and evaluation without dropout (learning_phase=false_
        results = sess.run(metrics_list, feed_dict={x: X, K.learning_phase(): False})
        results_str = "{:%Y-%m-%d-%H-%M}, {}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\n".format(
            datetime.datetime.now(), epoch + 1, *list(results)
        )
        with open(results_filename, "a") as fh_results:
            fh_results.write(results_str)


def save_posterior(sess, probs_tril_tf, posterior_param_tf, x, X, checkpoint_dir):
    posterior_param = sess.run(posterior_param_tf, feed_dict={x: X})
    probs_tril = sess.run(probs_tril_tf, feed_dict={x: X})

    with open(checkpoint_dir + "/posterior_0.pickle", "wb") as f:
        pickle.dump(posterior_param, f)

    with open(checkpoint_dir + "/prior.pickle", "wb") as f:
        pickle.dump(probs_tril, f)


def get_results_handler(results_dir, header):

    # setup writing results to disk
    results_filename = None
    if results_dir is not None:
        if not os.path.exists(os.path.expanduser(results_dir)):
            print("Results dir does not exist.")
            print("Creating results dir at {}".format(os.path.expanduser(results_dir)))
            os.makedirs(os.path.expanduser(results_dir))
            print(
                "Created results directory: {}".format(os.path.expanduser(results_dir))
            )
        else:
            print("Results directory already exists.")

        # write headers on results file
        results_filename = os.path.join(os.path.expanduser(results_dir), "results.csv")

        # write the column names
        with open(results_filename, "w", buffering=1) as fh_results:
            fh_results.write(header + "\n")

    return results_filename


def save_final_results(results_dir, results, predictions, y, mask_test):
    # output final results and predictions are finishing
    with open("bayesian_results.csv", "a+") as f:
        results_str = "{}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\n".format(
            results_dir, *list(results)
        )
        f.write(results_str)

    y_test_ = y[mask_test]
    y_pred_ = predictions[0][mask_test]
    y_pred_probs_ = predictions[0][mask_test]

    y_test_ = np.where(y_test_.flatten() > 0)[0] % y_test_.shape[1]
    y_pred_ = np.argmax(y_pred_, axis=1)

    print(confusion_matrix(y_test_, y_pred_))

    if results_dir is not None:  # Write y_test_ and y_pred_ to disk
        write_test_predictions(
            results_dir, y_true=y_test_, y_pred=y_pred_, y_pred_probs=y_pred_probs_
        )


def write_test_predictions(results_dir, y_true, y_pred, y_pred_probs):

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    for ind, col in enumerate(y_pred_probs.transpose()):
        df["y_pred_{}".format(ind)] = col

    predictions_filename = os.path.join(
        os.path.expanduser(results_dir), "predictions.csv"
    )
    df.to_csv(predictions_filename, index=None)


def save_parameters(results_dir, params):
    if results_dir is not None:
        if not os.path.exists(os.path.expanduser(results_dir)):
            print("Results dir does not exist.")
            print("Creating results dir at {}".format(os.path.expanduser(results_dir)))
            os.makedirs(os.path.expanduser(results_dir))
            print(
                "Created results directory: {}".format(os.path.expanduser(results_dir))
            )
        else:
            print("Results directory already exists.")

        # write parameters file
        params_filename = os.path.join(os.path.expanduser(results_dir), "params.csv")
        try:
            with open(params_filename, "w", buffering=1) as fh_params:
                w = csv.DictWriter(fh_params, params.keys())
                w.writeheader()
                w.writerow(params)

        except IOError:
            print("Could not open results file {}".format(params_filename))
    return


def safe_create_dir(results_dir):

    if results_dir is not None:
        if not os.path.exists(os.path.expanduser(results_dir)):
            print("Results dir does not exist.")
            print("Creating results dir at {}".format(os.path.expanduser(results_dir)))
            os.makedirs(os.path.expanduser(results_dir))
            print(
                "Created results directory: {}".format(os.path.expanduser(results_dir))
            )
        else:
            print("Results directory already exists.")


def get_final_results_cluster(
    sess, no_op, xs, Xs, final_metrics_list, y_pred, y, mask_test, results_dir
):
    # Calling no-op to trigger logging tensor hook to print metrics
    # in prediction phase.
    results = []
    predictions = []
    for x, X, final_metrics_list_, y_pred_ in zip(xs, Xs, final_metrics_list, y_pred):
        results.append(
            sess.run(final_metrics_list_, feed_dict={x: X, K.learning_phase(): False})
        )
        sess.run(
            no_op, feed_dict={x: X}
        )  # triggering Keras prediction phase (no Dropout)
        predictions.append(sess.run([y_pred_], feed_dict={x: X}))

    # TODO: Update the below line to work
    # save_final_results(results_dir, results, predictions, y, mask_test)


def get_intermediate_results_cluster(
    sess, metrics_list, xs, Xs, epoch, results_filename
):
    if (
        results_filename is not None
    ):  # predictions and evaluation without dropout (learning_phase=false_
        results = []
        for x, X, metrics_list_ in zip(xs, Xs, metrics_list):
            results.append(
                sess.run(metrics_list_, feed_dict={x: X, K.learning_phase(): False})
            )

        results = list(np.array(results).mean(axis=0))

        results_str = "{:%Y-%m-%d-%H-%M}, {}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\n".format(
            datetime.datetime.now(), epoch + 1, *list(results)
        )
        #results_str = "{}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\n".format(
        #    epoch + 1, *list(results)
        #)
        with open(results_filename, "a") as fh_results:
            fh_results.write(results_str)
