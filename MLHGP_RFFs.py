'''
Author: Edoardo Caldarelli
Affiliation: Institut de Robòtica i Informàtica Industrial, CSIC-UPC
email: ecaldarelli@iri.upc.edu
October 2023
'''

import numpy as np
import numpy.linalg
from scipy.linalg import cholesky, cho_solve
import gpflow
from gp_utilities import *
import matplotlib.pyplot as plt
import time
import tensorflow_probability as tfp

def mlhgp_rffs(Xs, X, Y, niter=20, num_rff=10, train=True, params_mean=None, params_variance=None, lth=.1, bounds=False, R=None, Rs=None):
    """
    This function trains a heteroscedastic GP with expectation-maximization.
    :param niter: number of iterations to be used in the training.
    :param bounds: whether to set bounds of the hyperparameters of the GPs.
    :param lth: initial value of the lengthscale.
    :param params_variance: hyperparameters for the GP on the noise variance.
    :param params_mean: hyperparameters for the GP on the mean function.
    :param train: whether the GP should be trained (True) or used for testing only (False).
    :param Xs: prediction inputs.
    :param X:  training inputs.
    :param Y:  training outputs.

    :return: Posterior mean, covariance and GP hyperparameters.
    """
    # Normalization of the observations.
    Ymean = np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Y = (Y - Ymean) / Ystd

    prediction_times = []
    optimization_times = []
    if train == False:
        # print("The hyperparameters will not be optimized")
        niter = 1

    # Step 1
    kernel = SquaredExponentialRFF(variance=1.0, lengthscales=lth, n_features=num_rff)
    model_gpflow = RFFGPR((X, Y), kernel=kernel)
    if train:
        old_parameter = model_gpflow.kernel.variance
        new_parameter = gpflow.Parameter(
            1e0,
            trainable=old_parameter.trainable,
            prior=old_parameter.prior,
            name=old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
            transform=tfp.bijectors.Sigmoid(np.float64(1e-2), np.float64(2)),
        )
        model_gpflow.kernel.variance = new_parameter

        old_parameter = model_gpflow.kernel.lengthscales
        new_parameter = gpflow.Parameter(
            1e-1,
            trainable=old_parameter.trainable,
            prior=old_parameter.prior,
            name=old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
            transform=tfp.bijectors.Sigmoid(np.float64(1e-2), np.float64(1e0)),
        )
        model_gpflow.kernel.lengthscales = new_parameter

        old_parameter = model_gpflow.likelihood.variance
        new_parameter = gpflow.Parameter(
            1e0,
            trainable=old_parameter.trainable,
            prior=old_parameter.prior,
            name=old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
            transform=tfp.bijectors.Sigmoid(np.float64(1e-2), np.float64(2)),
        )

        model_gpflow.likelihood.variance = new_parameter
        opt = gpflow.optimizers.Scipy()
        start_opti_time = time.time()
        opt.minimize(model_gpflow.training_loss, model_gpflow.trainable_variables)
        optimization_times.append(time.time() - start_opti_time)
    else:
        param = params_mean
        kernel.lengthscales = param[0]
        kernel.variance = param[1]
        model_gpflow.likelihood.variance = param[2]
    g1m, g1cov = model_gpflow.predict_f(X)
    g1m = g1m.numpy()
    g1cov = g1cov.numpy()
    # EM
    for i in range(0, niter):
        print("Iteration ", i)
        # Step 2
        r1 = 0.5 * ((Y.ravel() - g1m.ravel()) ** 2 + np.diag(g1cov).ravel())
        Z = np.log(r1).reshape(-1, 1)
        # Step 3
        kernel2 = SquaredExponentialRFF(variance=1.0, lengthscales=lth, n_features=num_rff)
        if train:
            model_gpflow2 = RFFGPR(data=(X, Z), kernel=kernel2)
            opt2 = gpflow.optimizers.Scipy()
            old_parameter = model_gpflow2.kernel.variance
            new_parameter = gpflow.Parameter(
                1e0,
                trainable=old_parameter.trainable,
                prior=old_parameter.prior,
                name=old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
                transform=tfp.bijectors.Sigmoid(np.float64(1e-1), np.float64(150)),
            )
            model_gpflow2.kernel.variance = new_parameter
            #
            old_parameter = model_gpflow2.kernel.lengthscales
            new_parameter = gpflow.Parameter(
                1e-1,
                trainable=old_parameter.trainable,
                prior=old_parameter.prior,
                name=old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
                transform=tfp.bijectors.Sigmoid(np.float64(1e-2), np.float64(1e0)),
            )

            model_gpflow2.kernel.lengthscales = new_parameter
            #
            old_parameter = model_gpflow2.likelihood.variance
            new_parameter = gpflow.Parameter(
                5e-1,
                trainable=old_parameter.trainable,
                prior=old_parameter.prior,
                name=old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
                transform=tfp.bijectors.Sigmoid(np.float64(1e-1), np.float64(1e0)),
            )

            model_gpflow2.likelihood.variance = new_parameter

            start_opti_time = time.time()
            opt2.minimize(model_gpflow2.training_loss, model_gpflow2.trainable_variables, options={'disp': 1})
            optimization_times.append(time.time() - start_opti_time)
            g2m, g2cov = model_gpflow2.predict_f(X)
            g2ms, g2covs = model_gpflow2.predict_f(Xs)

            g2m = g2m.numpy()
            g2cov = g2cov.numpy()
            g2ms = g2ms.numpy()
            g2covs = g2covs.numpy()

            # Step 4
            R = np.exp(g2m + g2cov / 2.0)
            Rs = np.exp(g2ms + g2covs / 2.0)
        else:
            model_gpflow2 = RFFGPR(data=(X, Z), kernel=kernel2)
            model_gpflow2.kernel.lengthscales = params_variance[0]
            model_gpflow2.kernel.variance = params_variance[1]
            model_gpflow2.likelihood.variance = params_variance[2]
            R = R
            Rs = Rs

        # phi = model_gpflow.kernel.compute_feature_vector(X).numpy()
        # phi_new = model_gpflow.kernel.compute_feature_vector(X).numpy() # Estimate variance at training inputs (smoothing)
        # jitter = 1e-8
        #
        # Rinv_vec = np.reciprocal(R + jitter)
        # Rinv = np.diag(Rinv_vec.ravel())
        # Rs = np.diag(Rs.ravel())
        # inner_matrix = np.eye(phi.shape[1]) + phi.T @ Rinv @ phi
        # L = np.linalg.cholesky(inner_matrix + jitter * np.eye(inner_matrix.shape[0]))
        #
        # Linv_phi = np.linalg.solve(L, phi.T @ Rinv)
        # mat_inv = Rinv - Linv_phi.T @ Linv_phi
        #
        # g1m = phi_new @ (phi.T @ (mat_inv @ Y))
        model_gpflow.likelihood.variance = R.ravel().reshape([-1, 1])
        g1m, g1cov = model_gpflow.predict_f(X, full_cov=False)
        g1m = g1m.numpy()
        g1cov = g1cov.numpy()
        # g1cov = phi_new @ phi_new.T + R - phi_new @ (phi.T @ (mat_inv @ (phi @ phi_new.T)))
        g1cov = g1cov + np.diag(R.ravel())

    # Final GP
    start_pred_time = time.time()
    # phi = model_gpflow.kernel.compute_feature_vector(X).numpy()
    # phi_new = model_gpflow.kernel.compute_feature_vector(Xs).numpy()  # Estimate variance at training inputs (smoothing)
    # inner_matrix = np.eye(phi.shape[1]) + phi.T @ Rinv @ phi
    # L = np.linalg.cholesky(inner_matrix)
    # Linv_phi = np.linalg.solve(L, phi.T @ Rinv)
    # mat_inv = Rinv - Linv_phi.T @ Linv_phi
    #
    # mu = phi_new @ (phi.T @ (mat_inv @ Y))
    # cov = phi_new @ phi_new.T + Rs - phi_new @ (phi.T @ (mat_inv @ (phi @ phi_new.T)))
    model_gpflow.likelihood.variance = R.ravel().reshape([-1, 1])
    mu, cov = model_gpflow.predict_f(Xs, full_cov=False)
    mu = mu.numpy()
    cov = cov.numpy()
    cov = cov + Rs
    mu = Ystd * mu + Ymean
    cov = cov * Ystd ** 2
    prediction_times.append(time.time() - start_pred_time)

    gp_var_params = [model_gpflow2.kernel.lengthscales, model_gpflow2.kernel.variance, model_gpflow2.likelihood.variance]
    gp_mean_params = [model_gpflow.kernel.lengthscales, model_gpflow.kernel.variance, model_gpflow.likelihood.variance]
    return mu, cov, gp_var_params, gp_mean_params, prediction_times, optimization_times, R, Rs


def mlhgp_nystrom(Xs, X, Y, niter=20, num_feat=100, train=True, params_mean=None, params_variance=None, lth=.1, bounds=False, R=None, Rs=None):
    """
    This function trains a heteroscedastic GP with expectation-maximization.
    :param niter: number of iterations to be used in the training.
    :param bounds: whether to set bounds of the hyperparameters of the GPs.
    :param lth: initial value of the lengthscale.
    :param params_variance: hyperparameters for the GP on the noise variance.
    :param params_mean: hyperparameters for the GP on the mean function.
    :param train: whether the GP should be trained (True) or used for testing only (False).
    :param Xs: prediction inputs.
    :param X:  training inputs.
    :param Y:  training outputs.

    :return: Posterior mean, covariance and GP hyperparameters.
    """
    # Normalization of the observations.
    Ymean = np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Y = (Y - Ymean) / Ystd

    prediction_times = []
    optimization_times = []
    # Step 1
    nystrom_centers = np.random.choice(X.squeeze(), (num_feat, 1), replace=False)
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=lth)
    model_gpflow = NystromGPR(data=(X, Y), kernel=kernel, nystrom_centers=nystrom_centers)
    if train:
        opt = gpflow.optimizers.Scipy()
        print('starting opt...')
        start_opti_time = time.time()
        opt.minimize(model_gpflow.training_loss, model_gpflow.trainable_variables)
        optimization_times.append(time.time() - start_opti_time)
    else:
        param = params_mean
        kernel.lengthscales = param[0]
        kernel.variance = param[1]
        model_gpflow.likelihood.variance = param[2]
    g1m, g1cov = model_gpflow.predict_f(X)
    g1m = g1m.numpy()
    g1cov = g1cov.numpy()
    # EM
    kernel2 = gpflow.kernels.RBF(variance=1.0, lengthscales=lth)
    for i in range(0, niter):
        print("Iteration ", i)
        # Step 2
        r1 = 0.5 * ((Y.ravel() - g1m.ravel()) ** 2 + np.diag(g1cov).ravel())
        Z = np.log(r1).reshape(-1, 1)
        # Step 3
        model_gpflow2 = NystromGPR(data=(X, Z), kernel=kernel2, nystrom_centers=nystrom_centers)
        if train:
            opt2 = gpflow.optimizers.Scipy()
            start_opti_time = time.time()
            opt2.minimize(model_gpflow2.training_loss, model_gpflow2.trainable_variables)
            optimization_times.append(time.time() - start_opti_time)
            g2m, g2cov = model_gpflow2.predict_f(X)
            g2ms, g2covs = model_gpflow2.predict_f(Xs)

            g2m = g2m.numpy()
            g2cov = g2cov.numpy()
            g2ms = g2ms.numpy()
            g2covs = g2covs.numpy()

            # Step 4
            R = np.exp(g2m + g2cov / 2.0)
            Rs = np.exp(g2ms + g2covs / 2.0)
        else:
            model_gpflow2.kernel.lengthscales = params_variance[0]
            model_gpflow2.kernel.variance = params_variance[1]
            model_gpflow2.likelihood.variance = params_variance[2]
            R = R
            Rs = Rs
            # g2m, g2cov = model_gpflow2.predict_f(X)
            # g2ms, g2covs = model_gpflow2.predict_f(Xs)
            #
            # g2m = g2m.numpy()
            # g2cov = g2cov.numpy()
            # g2ms = g2ms.numpy()
            # g2covs = g2covs.numpy()
            #
            # # Step 4
            # R = np.exp(g2m + g2cov / 2.0)
            # Rs = np.exp(g2ms + g2covs / 2.0)

        K = kernel.K(X, X).numpy() + np.diag(R.ravel()) + 1e-8 * np.eye(len(X))
        Ks = kernel.K(X, X).numpy()
        Kss = kernel.K(X, X).numpy() + np.diag(R.ravel())

        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), Y)
        v = cho_solve((L, True), Ks.T)

        g1m = Ks.dot(alpha)
        g1cov = Kss - Ks.dot(v)

    # Final GP
    start_pred_time = time.time()
    K = kernel.K(X, X).numpy() + np.diag(R.ravel()) + 1e-8 * np.eye(len(X))
    Ks = kernel.K(Xs, X).numpy()
    Kss = kernel.K(Xs, Xs).numpy() + np.diag(Rs.ravel())

    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), Y)
    v = cho_solve((L, True), Ks.T)

    mu = Ks.dot(alpha)
    cov = Kss - Ks.dot(v)

    mu = Ystd * mu + Ymean
    cov = cov * Ystd ** 2
    prediction_times.append(time.time() - start_pred_time)

    gp_var_params = [model_gpflow2.kernel.lengthscales, model_gpflow2.kernel.variance, model_gpflow2.likelihood.variance]
    gp_mean_params = [model_gpflow.kernel.lengthscales, model_gpflow.kernel.variance, model_gpflow.likelihood.variance]
    return mu, np.diag(cov), gp_var_params, gp_mean_params, prediction_times, optimization_times, R, Rs

def mlhgp(Xs, X, Y, niter=20, train=True, params_mean=None, params_variance=None, lth=.1, bounds=False, R=None, Rs=None):
    """
    This function trains a heteroscedastic GP with expectation-maximization.
    :param niter: number of iterations to be used in the training.
    :param bounds: whether to set bounds of the hyperparameters of the GPs.
    :param lth: initial value of the lengthscale.
    :param params_variance: hyperparameters for the GP on the noise variance.
    :param params_mean: hyperparameters for the GP on the mean function.
    :param train: whether the GP should be trained (True) or used for testing only (False).
    :param Xs: prediction inputs.
    :param X:  training inputs.
    :param Y:  training outputs.

    :return: Posterior mean, covariance and GP hyperparameters.
    """
    # Normalization of the observations.
    Ymean = np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Y = (Y - Ymean) / Ystd

    prediction_times = []
    optimization_times = []
    # Step 1
    kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=lth)
    model_gpflow = gpflow.models.GPR((X, Y), kernel=kernel)
    if train:
        opt = gpflow.optimizers.Scipy()
        print('starting opt...')
        start_opti_time = time.time()
        opt.minimize(model_gpflow.training_loss, model_gpflow.trainable_variables)
        optimization_times.append(time.time() - start_opti_time)
    else:
        param = params_mean
        kernel.lengthscales = param[0]
        kernel.variance = param[1]
        model_gpflow.likelihood.variance = param[2]
    g1m, g1cov = model_gpflow.predict_f(X)
    g1m = g1m.numpy()
    g1cov = g1cov.numpy()
    # EM
    kernel2 = gpflow.kernels.RBF(variance=1.0, lengthscales=lth)
    for i in range(0, niter):
        print("Iteration ", i)
        # Step 2
        r1 = 0.5 * ((Y.ravel() - g1m.ravel()) ** 2 + np.diag(g1cov).ravel())
        Z = np.log(r1).reshape(-1, 1)
        # Step 3
        model_gpflow2 = gpflow.models.GPR(data=(X, Z), kernel=kernel2)
        if train:
            opt2 = gpflow.optimizers.Scipy()
            start_opti_time = time.time()
            opt2.minimize(model_gpflow2.training_loss, model_gpflow2.trainable_variables)
            optimization_times.append(time.time() - start_opti_time)
            g2m, g2cov = model_gpflow2.predict_f(X)
            g2ms, g2covs = model_gpflow2.predict_f(Xs)

            g2m = g2m.numpy()
            g2cov = g2cov.numpy()
            g2ms = g2ms.numpy()
            g2covs = g2covs.numpy()

            # Step 4
            R = np.exp(g2m + g2cov / 2.0)
            Rs = np.exp(g2ms + g2covs / 2.0)
        else:
            model_gpflow2.kernel.lengthscales = params_variance[0]
            model_gpflow2.kernel.variance = params_variance[1]
            model_gpflow2.likelihood.variance = params_variance[2]
            R = R
            Rs = Rs
            # g2m, g2cov = model_gpflow2.predict_f(X)
            # g2ms, g2covs = model_gpflow2.predict_f(Xs)
            #
            # g2m = g2m.numpy()
            # g2cov = g2cov.numpy()
            # g2ms = g2ms.numpy()
            # g2covs = g2covs.numpy()
            #
            # # Step 4
            # R = np.exp(g2m + g2cov / 2.0)
            # Rs = np.exp(g2ms + g2covs / 2.0)

        K = kernel.K(X, X).numpy() + np.diag(R.ravel()) + 1e-8 * np.eye(len(X))
        Ks = kernel.K(X, X).numpy()
        Kss = kernel.K(X, X).numpy() + np.diag(R.ravel())

        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), Y)
        v = cho_solve((L, True), Ks.T)

        g1m = Ks.dot(alpha)
        g1cov = Kss - Ks.dot(v)

    # Final GP
    start_pred_time = time.time()
    K = kernel.K(X, X).numpy() + np.diag(R.ravel()) + 1e-8 * np.eye(len(X))
    Ks = kernel.K(Xs, X).numpy()
    Kss = kernel.K(Xs, Xs).numpy() + np.diag(Rs.ravel())

    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), Y)
    v = cho_solve((L, True), Ks.T)

    mu = Ks.dot(alpha)
    cov = Kss - Ks.dot(v)

    mu = Ystd * mu + Ymean
    cov = cov * Ystd ** 2
    prediction_times.append(time.time() - start_pred_time)

    gp_var_params = [model_gpflow2.kernel.lengthscales, model_gpflow2.kernel.variance, model_gpflow2.likelihood.variance]
    gp_mean_params = [model_gpflow.kernel.lengthscales, model_gpflow.kernel.variance, model_gpflow.likelihood.variance]
    return mu, np.diag(cov), gp_var_params, gp_mean_params, prediction_times, optimization_times, R, Rs