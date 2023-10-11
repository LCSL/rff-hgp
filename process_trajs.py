'''
Author: Edoardo Caldarelli
Affiliation: Institut de Robòtica i Informàtica Industrial, CSIC-UPC
email: ecaldarelli@iri.upc.edu
October 2023
'''

"""
This script processes the DOFs of a trajectory with heteroscedastic GP regression and random Fourier features.
"""

import argparse
import numpy as np
import pathlib
from scipy.spatial.distance import euclidean
from dtw import accelerated_dtw
import pickle
import random
import tensorflow as tf

from MLHGP_RFFs import mlhgp, mlhgp_rffs
from MLHGP_variational import mlhgp_svgp
import pandas as pd


def align_trajectories(trajs, given_template):
    """
    This method aligns the series by means of dynamic time warping (DTW).
    :param trajs: list of trajectories (one element per demonstration).
    :param given_template: the reference to be used in the alignment
    :return: a list of aligned trajectories (one element is the union of the observations of a given DOF, across the
    demonstrations. The indices of the observations are also returned.
    """
    template = given_template.copy()
    warped_trajs = []
    indices = []
    for i in range(0, len(trajs) - 1):

        print("Aligning trial", i, "...")

        query = trajs[i]
        _, _, _, aligned = accelerated_dtw(template, query, dist=euclidean)

        warped = query[aligned[1]]

        tw = template[aligned[0]]

        if i == len(trajs) - 2:
            warped_trajs.append(tw)
            indices.append(np.arange(0, tw.shape[0], 1) / tw.shape[0])

        warped_trajs.append(warped)
        indices.append(np.arange(0, warped.shape[0], 1) / warped.shape[0])

    trajs_union = []
    indices_union = []

    for dim in range(0, num_dofs):
        curr_union_trajs = np.empty(0)
        curr_union_indices = np.empty(0)
        for trial_index, trial in enumerate(warped_trajs):
            curr_traj = trial[:, dim]
            curr_union_trajs = np.concatenate((curr_union_trajs, curr_traj))
            curr_union_indices = np.concatenate((curr_union_indices, indices[trial_index]))

        trajs_union.append(curr_union_trajs)
        indices_union.append(curr_union_indices)

    return indices_union, trajs_union


def sort_time_series(indices, outputs):
    """
    This function sorts the indices and corresponding observations of a DOF. Typically, these observations result from a
    union across multiple demonstrations.
    :param indices: the indices of the DOFs.
    :param outputs: the observations to be sorted.
    :return: the sorted indices and observations.
    """
    times, funcs = [], []
    for i, dof in enumerate(outputs):
        full_dataset = np.concatenate((indices[i].reshape([-1, 1]), outputs[i].reshape([-1, 1])), axis=-1)  # [::10, :]
        full_sorted_dataset = full_dataset[np.argsort(full_dataset[:, 0])]
        times.append(full_sorted_dataset[:, 0])
        funcs.append(full_sorted_dataset[:, 1:])
    return times, funcs


def process_MOGP(indices_union, series_union, x_test=None, gp_type='exact', train_exact_GP = False,
                 train_rff_GP=False, pretrained_gp_mean_params=None, pretrained_gp_var_params=None,
                 n_rffs=100, pretrained_R=None, pretrained_Rs=None):
    """
    This function processes a set of time series with heteroscedastic GP regression. Each DOF is processed independently
    of the others.
    :param indices_union: list of the indices of the trajectories (one element per DOF).
    :param series_union: list of observations (one element per DOF).
    :param x_test: time-steps at whoch the posterior distribution is needed. If None, the same indices passed as first
    parameter are used.
    :return: the posterior means and posterior covariances of the GP (in a list, one element per DOF), the testing
    inputs, and the parameters on the GPs for the noise variance and the function's mean.
    """
    preds = []
    uncs = []
    x_tests = []
    gp_vars_params = []
    gp_means_params = []
    prediction_times = []
    optimization_times = []
    R_list = []
    Rs_list = []

    for dof_indx in range(0, len(series_union)):

        X = indices_union[dof_indx].reshape([-1, 1])
        Y = series_union[dof_indx].reshape([-1, 1])

        if x_test is None:
            x_test = X
        if gp_type == 'exact':
            niter = 1 if not train_exact_GP else 15
            if not train_exact_GP:
                pretr_mean_par = pretrained_gp_mean_params[dof_indx]
                pretr_var_par = pretrained_gp_var_params[dof_indx]
                pretr_R = pretrained_R[dof_indx]
                pretr_Rs = pretrained_Rs[dof_indx]
            else:
                pretr_mean_par = None
                pretr_var_par = None
                pretr_R = None
                pretr_Rs = None
            curr_pred, curr_unc, gp_var_params, gp_mean_params, curr_prediction_times, curr_optimization_times, R, Rs = \
                mlhgp(x_test, X, Y, train=train_exact_GP,
                      params_mean=pretr_mean_par,
                      params_variance=pretr_var_par, R=pretr_R, Rs=pretr_Rs,
                      niter=niter)
        elif gp_type == 'rffs':
            niter = 1 if not train_rff_GP else 15
            if not train_rff_GP:
                pretr_mean_par = pretrained_gp_mean_params[dof_indx]
                pretr_var_par = pretrained_gp_var_params[dof_indx]
                pretr_R = pretrained_R[dof_indx]
                pretr_Rs = pretrained_Rs[dof_indx]
            else:
                pretr_mean_par = None
                pretr_var_par = None
                pretr_R = None
                pretr_Rs = None

            curr_pred, curr_unc, gp_var_params, gp_mean_params, curr_prediction_times, curr_optimization_times, R, Rs = \
                mlhgp_rffs(x_test, X, Y,
                           niter=niter,
                           train=train_rff_GP,
                           params_mean=pretr_mean_par,
                           params_variance=pretr_var_par,
                           num_rff=n_rffs,
                           R=pretr_R,
                           Rs=pretr_Rs)
        elif gp_type == "svgp":
            niter = 1 if not train_rff_GP else 15
            if not train_rff_GP:
                pretr_mean_par = pretrained_gp_mean_params[dof_indx]
                pretr_var_par = pretrained_gp_var_params[dof_indx]
                pretr_R = pretrained_R[dof_indx]
                pretr_Rs = pretrained_Rs[dof_indx]
            else:
                pretr_mean_par = None
                pretr_var_par = None
                pretr_R = None
                pretr_Rs = None

            curr_pred, curr_unc, gp_var_params, gp_mean_params, curr_prediction_times, curr_optimization_times, R, Rs = \
                mlhgp_svgp(x_test, X, Y,
                           niter=niter,
                           train=train_rff_GP,
                           params_mean=pretr_mean_par,
                           params_variance=pretr_var_par,
                           num_feat=n_rffs,
                           R=pretr_R,
                           Rs=pretr_Rs)
        else:
            print('Wrong GP type!')
            exit(1)

        preds.append(curr_pred)
        uncs.append(curr_unc)

        x_tests.append(x_test)

        gp_vars_params.append(gp_var_params)
        gp_means_params.append(gp_mean_params)

        prediction_times.append(curr_prediction_times)
        optimization_times.append(curr_optimization_times)
        R_list.append(R)
        Rs_list.append(Rs)

    return preds, uncs, x_tests, gp_vars_params, gp_means_params, prediction_times, optimization_times, R_list, Rs_list


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    num_dofs = 6  # number of DOFs dimensions
    for experiment in ["proof-of-concept", "assembly-task", "bed-making"]:
        if experiment == "proof-of-concept":
            n_trials_unperturbed = 6
            time_horizon = 20
        elif experiment == "assembly-task":
            n_trials_unperturbed = 7
            time_horizon = 30
        else:
            n_trials_unperturbed = 5
            time_horizon = 30
        path_to_trajs_unperturbed = pathlib.Path(f"{experiment}/unperturbed/trajs")

        disturbances = np.empty((0, 1))
        indices_disturbance = np.empty((0, 1))

        trajs = []
        for i in range(0, n_trials_unperturbed):
            trajs_u_df = pd.read_csv(f"{path_to_trajs_unperturbed}/CartesianQ{i}.txt", header=None, usecols=range(0, num_dofs))
            trajs.append(trajs_u_df.to_numpy(dtype=np.float32))

        print("------------ Processing trajectories...")
        given_template = trajs[-1]

        time_union_trajs, trajs_union = align_trajectories(trajs, given_template)

        with open(f"{experiment}/indices_union.txt", "wb") as fp:
            pickle.dump(time_union_trajs, fp)

        with open(f"{experiment}/trajs_union.txt", "wb") as fp:
            pickle.dump(trajs_union, fp)
        sorted_indices, sorted_trajs = sort_time_series(time_union_trajs, trajs_union)
        with open(f"{experiment}/sorted_indices.txt", "wb") as fp:
            pickle.dump(sorted_indices, fp)

        gp_types = ['exact', 'rffs', 'svgp']
        gp_means_params = None
        gp_vars_params = None
        R_list = None
        Rs_list = None
        train_rff_GP = False
        train_exact_GP = True
        for gp_type in gp_types:
            seeds = [0] if gp_type == 'exact' else np.arange(0, 50)
            ns_rffs = [0] if gp_type == 'exact' else np.arange(10, 250, 10)
            for n_rffs in ns_rffs:
                for seed in seeds:
                    np.random.seed(seed)
                    random.seed(seed)
                    tf.random.set_seed(seed)
                    print("N RFFs ", n_rffs, ", SEED ", seed)
                    # Train a GP to get the reference signals to be followed along each DOF.
                    x_test = np.arange(0, 1, 0.5e-3).reshape([-1, 1])
                    if not train_rff_GP or not train_exact_GP:
                        pretrained_mean_params = gp_means_params
                        pretrained_var_params = gp_vars_params
                        pretrained_R = R_list
                        pretrained_Rs = Rs_list
                    else:
                        pretrained_mean_params = None
                        pretrained_var_params = None
                        pretrained_R = None
                        pretrained_Rs = None
                    reference_signals, uncertainties, x_tests, \
                        gp_vars_params, gp_means_params, prediction_times, optimization_times, R_list, Rs_list = process_MOGP(sorted_indices,
                                                                                                              sorted_trajs,
                                                                                                              x_test,
                                                                                                              gp_type,
                                                                                                              train_exact_GP,
                                                                                                              train_rff_GP,
                                                                                                              pretrained_mean_params,
                                                                                                              pretrained_var_params,
                                                                                                              n_rffs,
                                                                                                              pretrained_R,
                                                                                                              pretrained_Rs)
                    assert len(uncertainties) == 6 and uncertainties[0].shape[0] == 2000

                    with open(f"{experiment}/reference_signals_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt", "wb") as fp:
                        pickle.dump(reference_signals, fp)

                    with open(f"{experiment}/uncertainties_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt", "wb") as fp:
                        pickle.dump(uncertainties, fp)

                    with open(f"{experiment}/x_tests_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt", "wb") as fp:
                        pickle.dump(x_tests, fp)

                    with open(f"{experiment}/gp_vars_params_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt", "wb") as fp:
                        pickle.dump(gp_vars_params, fp)

                    with open(f"{experiment}/gp_means_params_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt", "wb") as fp:
                        pickle.dump(gp_means_params, fp)

                    with open(f"{experiment}/pred_times_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt", "wb") as fp:
                        pickle.dump(prediction_times, fp)

                    with open(f"{experiment}/opti_times_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt", "wb") as fp:
                        pickle.dump(optimization_times, fp)

                    with open(f"{experiment}/R_list_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt",
                            "wb") as fp:
                        pickle.dump(R_list, fp)

                    with open(f"{experiment}/Rs_list_gp_{gp_type}_seed_{seed}_train_rff_GP_{train_rff_GP}_n_rffs_{n_rffs}.txt",
                            "wb") as fp:
                        pickle.dump(Rs_list, fp)
