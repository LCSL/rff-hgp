'''
Author: Edoardo Caldarelli
Affiliation: Institut de Robòtica i Informàtica Industrial, CSIC-UPC
email: ecaldarelli@iri.upc.edu
October 2023
'''

"""
This script processes the DOFs of a trajectory with heteroscedastic GP regression and random Fourier features. The
learning algorithm is incremental.
"""

import argparse
import copy

import numpy as np
import pathlib

import scipy
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from dtw import accelerated_dtw
import pickle
import random
import tensorflow as tf

from MLHGP_RFFs import mlhgp, mlhgp_rffs
import pandas as pd
from gp_utilities import *


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
        curr_union_trajs = []
        curr_union_indices = []
        for trial_index, trial in enumerate(warped_trajs):
            curr_traj = trial[:, dim]
            curr_union_trajs.append(curr_traj)
            curr_union_indices.append(indices[trial_index])

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


def process_MOGP(indices_union, series_union, x_test=None, gp_type='exact', train_rff_GP=False,
                 pretrained_gp_mean_params=None, pretrained_gp_var_params=None,
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
            curr_pred, curr_unc, gp_var_params, gp_mean_params, curr_prediction_times, curr_optimization_times, R, Rs = \
                mlhgp(x_test, X, Y, niter=15)
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

            try:
                curr_pred, curr_unc, gp_var_params, gp_mean_params, curr_prediction_times, curr_optimization_times, R, Rs = \
                    mlhgp_rffs(x_test, X, Y,
                               niter=niter,
                               train=train_rff_GP,
                               params_mean=pretr_mean_par,
                               params_variance=pretr_var_par,
                               num_rff=n_rffs,
                               R=pretr_R,
                               Rs=pretr_Rs)
            except Exception:
                print("The HGP fit has failed!")
                continue
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


def low_rank_update(R, phi_x_new, chol_fact=None):
    n_features = phi_x_new.shape[1]
    if chol_fact is None:
        chol_fact = np.eye(n_features)  # Initialize the Gram matrix with the identity
    last_rows = phi_x_new.T @ np.sqrt(R)
    chol_fact_tilde = np.vstack((chol_fact, last_rows.T))
    chol_fact = scipy.linalg.cholesky(chol_fact_tilde.T @ chol_fact_tilde,
                                      lower=False)  # chol_fact_tilde[: n_features, :]

    return chol_fact


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
        num_joints = 7  # Number of manipulator's joints. Needed for formatting.

        path_to_trajs_unperturbed = pathlib.Path(f"{experiment}/unperturbed/trajs")

        disturbances = np.empty((0, 1))
        indices_disturbance = np.empty((0, 1))

        trajs = []
        for dof in range(0, n_trials_unperturbed):
            trajs_u_df = pd.read_csv(f"{path_to_trajs_unperturbed}/CartesianQ{dof}.txt", header=None,
                                     usecols=range(0, num_dofs))
            trajs.append(trajs_u_df.to_numpy(dtype=np.float32))

        print("------------ Processing trajectories...")
        with open(f"{experiment}/sorted_indices.txt", "rb") as fp:
            sorted_indices = pickle.load(fp)
        given_template = trajs[-1]
        seeds = np.arange(0, 50)
        dims_feat_vector = np.arange(10, 100, 5)
        gp_types = ['rffs', 'nystrom']
        for gp_type in gp_types:
            with open(
                    f"{experiment}/gp_means_params_gp_exact_seed_0_train_rff_GP_False_n_rffs_0.txt",
                    "rb") as fp:
                gp_means_params = pickle.load(fp)

            with open(f"{experiment}/R_list_gp_exact_seed_0_train_rff_GP_False_n_rffs_0.txt",
                      "rb") as fp:
                R_list = pickle.load(fp)

            with open(f"{experiment}/Rs_list_gp_exact_seed_0_train_rff_GP_False_n_rffs_0.txt",
                      "rb") as fp:
                Rs_list = pickle.load(fp)
            with open(f"{experiment}/reference_signals_gp_exact_seed_0_train_rff_GP_False_n_rffs_0.txt",
                      "rb") as fp:
                exact_reference_list = pickle.load(fp)

            with open(f"{experiment}/uncertainties_gp_exact_seed_0_train_rff_GP_False_n_rffs_0.txt",
                      "rb") as fp:
                exact_cov_list = pickle.load(fp)
            x_test = np.arange(0, 1, 0.5e-3).reshape([-1, 1])
            times_per_dof, trajs_per_dof = align_trajectories(trajs, given_template)
            for dim_feat_vector in dims_feat_vector:
                for seed in seeds:
                    np.random.seed(seed)
                    random.seed(seed)
                    tf.random.set_seed(seed)
                    print("DIM feat vec", dim_feat_vector, " SEED ", seed, " Method ", gp_type, " Experiment ", experiment)

                    reference_signals = []
                    uncertainties = []
                    R_per_dof = []
                    for dof, trials in enumerate(trajs_per_dof):
                        R_current_dof = R_list[dof]
                        R_per_trial = []
                        for trial_index, trial in enumerate(trials):
                            indices_first_array = np.where(np.isin(sorted_indices[dof], times_per_dof[dof][trial_index]))[0]
                            filtered_R = np.unique(R_current_dof[indices_first_array])
                            if filtered_R.shape[0] > trial.shape[0]:
                                filtered_R = filtered_R[:-1]
                            R_curr = np.diag(np.reciprocal(filtered_R + 1e-4).squeeze())
                            R_per_trial.append(R_curr)
                        R_per_dof.append(R_per_trial)

                    sliced_times_per_dof = copy.deepcopy(times_per_dof)
                    sliced_trajs_per_dof = copy.deepcopy(trajs_per_dof)
                    sliced_R_per_dof = copy.deepcopy(R_per_dof)

                    for dof, sliced_trials in enumerate(sliced_trajs_per_dof):
                        # Remove portion of the trajectory
                        for trial_index, _ in enumerate(sliced_trials):
                            chunk_size = 60
                            indx = np.random.randint(0, len(sliced_trials[trial_index]) - chunk_size)
                            chunk = np.arange(indx, indx + chunk_size)
                            mask = np.ones(len(times_per_dof[dof][trial_index]), dtype=bool)
                            mask[chunk] = False
                            sliced_times_per_dof[dof][trial_index] = copy.deepcopy(times_per_dof[dof][trial_index][mask])
                            sliced_trials[trial_index] = copy.deepcopy(trajs_per_dof[dof][trial_index][mask])
                            sliced_R_per_dof[dof][trial_index] = copy.deepcopy(R_per_dof[dof][trial_index][mask, :])
                            sliced_R_per_dof[dof][trial_index] = copy.deepcopy(sliced_R_per_dof[dof][trial_index][:, mask])

                        # Incremental updates
                        chol_fact = None
                        Phi = np.empty((0, dim_feat_vector))
                        Y = np.empty((0, 1))
                        Y_full = np.empty((0, 1))
                        if gp_type == 'rffs':
                            kernel = SquaredExponentialRFF(variance=gp_means_params[dof][1].numpy(),
                                                           lengthscales=gp_means_params[dof][0].numpy(),
                                                           n_features=dim_feat_vector)
                        elif gp_type == 'nystrom':
                            nystrom_centers = np.random.choice(sliced_times_per_dof[dof][0], size=dim_feat_vector, replace=False).reshape([-1, 1])
                            exact_kernel = gpflow.kernels.RBF(variance=gp_means_params[dof][1].numpy(),
                                                              lengthscales=gp_means_params[dof][0].numpy())
                            kernel = SquaredExponentialNystrom(nystrom_centers=nystrom_centers, exact_kernel=exact_kernel)
                        R_current_dof = np.empty(0)
                        A = np.zeros((dim_feat_vector, 1))
                        B = np.zeros((dim_feat_vector, 1))

                        for trial_index, trial in enumerate(sliced_trials):
                            Y_full = np.vstack((Y_full, trajs_per_dof[dof][trial_index].reshape([-1, 1])))

                        Ymean = np.mean(Y_full, axis=0)
                        Ystd = np.std(Y_full, axis=0)
                        W_mean = None
                        W_variance = None

                        if gp_type == 'nystrom' or gp_type == 'rffs':
                            phi_xtest = kernel.compute_feature_vector(x_test).numpy()
                            for trial_index, trial in enumerate(sliced_trials):
                                R_current_trial = sliced_R_per_dof[dof][trial_index]
                                phi_xnew = kernel.compute_feature_vector(sliced_times_per_dof[dof][trial_index].reshape([-1, 1])).numpy()
                                chol_fact = low_rank_update(R_current_trial, phi_xnew, chol_fact)
                                trial_std = (trial.reshape([-1, 1]) - Ymean) / Ystd
                                A = A + phi_xnew.T @ R_current_trial @ trial_std
                                B = B + phi_xnew.T @ R_current_trial @ phi_xnew

                                W_mean = A - np.linalg.solve(chol_fact.T, B.T).T @ np.linalg.solve(chol_fact.T, A)
                                W_variance = B - np.linalg.solve(chol_fact.T, B.T).T @ np.linalg.solve(chol_fact.T, B)

                                R_current_dof = np.concatenate((R_current_dof, np.diag(R_current_trial).ravel()))
                            mu = phi_xtest @ W_mean
                            sigma_test = phi_xtest @ phi_xtest.T if gp_type == "rffs" else kernel.exact_kernel.variance.numpy()

                            cov = sigma_test + np.diag(Rs_list[dof].ravel()) - phi_xtest @ (W_variance @ phi_xtest.T)
                            mu = Ystd * mu + Ymean
                            cov = cov * Ystd ** 2
                            cov = np.diag(cov)
                        else:
                            partial_X = np.empty((0, 1))
                            partial_Y = np.empty((0, 1))
                            for trial_index, trial in enumerate(sliced_trials):
                                R_current_trial = sliced_R_per_dof[dof][trial_index]
                                partial_X = np.vstack((partial_X, sliced_times_per_dof[dof][trial_index].reshape([-1, 1])))
                                trial_std = (trial.reshape([-1, 1]) - Ymean) / Ystd
                                partial_Y = np.vstack((partial_Y, trial_std))
                                R_current_dof = np.concatenate((R_current_dof, np.reciprocal(np.diag(R_current_trial).ravel())))

                            gp = gpflow.models.SGPR(data=(partial_X, partial_Y),
                                                    kernel=gpflow.kernels.RBF(lengthscales=gp_means_params[dof][0].numpy(),
                                                                       variance=gp_means_params[dof][1].numpy()),
                                                    inducing_variable=np.arange(0, dim_feat_vector).reshape([-1, 1]) / dim_feat_vector)

                            gp.likelihood.variance = R_current_dof.reshape([-1, 1])
                            mu, cov = gp.predict_f(x_test)
                            mu = mu.numpy()
                            cov = np.diag(cov.numpy()) + Rs_list[dof].ravel()
                            mu = Ystd * mu + Ymean
                            cov = cov * Ystd ** 2
                        reference_signals.append(mu)
                        uncertainties.append(cov)
                    with open(
                            f"{experiment}/reference_signals_gp_{gp_type}_seed_{seed}_incremental_n_feat_{dim_feat_vector}.txt",
                            "wb") as fp:
                        pickle.dump(reference_signals, fp)

                    with open(
                            f"{experiment}/uncertainties_gp_{gp_type}_seed_{seed}_incremental_n_feat_{dim_feat_vector}.txt",
                            "wb") as fp:
                        pickle.dump(uncertainties, fp)
