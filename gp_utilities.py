'''
Author: Edoardo Caldarelli
Affiliation: Institut de Robòtica i Informàtica Industrial, CSIC-UPC
email: ecaldarelli@iri.upc.edu
October 2023
'''

from check_shapes import check_shapes, inherit_check_shapes

from gpflow.base import InputData, MeanAndVariance, RegressionData, TensorData, TensorType, Parameter
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import MeanFunction
from gpflow.utilities import assert_params_false, positive
import tensorflow as tf
import gpflow
from gpflow.kernels.base import ActiveDims
import numpy as np
from typing import Optional

class SquaredExponentialRFF(gpflow.kernels.Kernel):
    @check_shapes(
        "variance: [broadcast n_active_dims]",
    )
    def __init__(
            self, variance: TensorType = 1.0, lengthscales: TensorType = 1.0, n_features=100, active_dims: Optional[ActiveDims] = None
    ) -> None:
        """
        :param variance: the (initial) value for the variance parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the number of active dimensions e.g. [1., 1., 1.]
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive(), name='kernel_var')
        self.lengthscales = Parameter(lengthscales, transform=positive(), name='kernel_lth')
        self.n_features = n_features
        self._validate_ard_active_dims(self.variance)
        input_dim = 1 if self.lengthscales.shape.ndims == 0 else self.lengthscales.shape.ndims
        omegas = tf.random.normal(shape=[input_dim, 1], dtype=tf.float64)
        for i in range(1, n_features):
            curr_omega = tf.random.normal(shape=[input_dim, 1], dtype=tf.float64)
            omegas = tf.concat([omegas, curr_omega], axis=-1)
        self.bias = tf.random.uniform(shape=[1, self.n_features], minval=np.float64(0), maxval=np.float64(2 * np.pi), dtype=tf.float64)
        self.omegas = omegas

    def compute_feature_vector(self, X: TensorType) -> TensorType:
        return tf.math.sqrt(2 * self.variance) * tf.sqrt(tf.math.reciprocal(tf.convert_to_tensor(self.n_features, dtype=tf.float64))) \
               * tf.math.cos(X @ (self.omegas / self.lengthscales) + self.bias)

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        if X2 is None:
            return self.compute_feature_vector(X) @ tf.transpose(self.compute_feature_vector(X))
        else:
            return self.compute_feature_vector(X) @ tf.transpose(self.compute_feature_vector(X2))

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        return tf.linalg.diag_part(self.K(X, X))

class RFFGPR(gpflow.models.GPR):
    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "noise_variance: []",
    )
    def __init__(self, data: RegressionData,
                 kernel: SquaredExponentialRFF,
                 mean_function: Optional[MeanFunction] = None,
                 noise_variance: Optional[TensorData] = None,
                 likelihood: Optional[Gaussian] = None,):
        super().__init__(data, kernel, mean_function, noise_variance, likelihood)

    @check_shapes(
        "return: []",
    )
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        phi = self.kernel.compute_feature_vector(X)
        sigma_2 = self.likelihood.variance_at(X)
        sigma = tf.squeeze(tf.math.sqrt(sigma_2), axis=-1)
        phi = tf.transpose(tf.transpose(phi) / sigma)
        # regularized_inverse_id = tf.linalg.tensor_diag(tf.math.reciprocal(sigma_2))
        inner_mat = tf.eye(phi.shape[1], dtype=tf.float64) + tf.transpose(phi) @ phi
        L = tf.linalg.cholesky(inner_mat)
        v = (Y - self.mean_function(X)) / sigma[..., None]
        phiTv = tf.transpose(phi) @ v

        L_invphiTv = tf.linalg.triangular_solve(L, phiTv, lower=True)
        # mat_inv = regularized_inverse_id - tf.transpose(Lphi_inv) @ Lphi_inv  # GP is homoscedastic
        log_prob = - 0.5 * tf.reduce_sum(tf.square(v)) \
                   + 0.5 * tf.reduce_sum(tf.square(L_invphiTv)) \
                   - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L))) \
                   - tf.reduce_sum(tf.math.log(sigma)) \
                   - 0.5 * Y.shape[0] * tf.math.log(2 * np.float64(np.pi))
        return tf.reduce_sum(log_prob)

    @inherit_check_shapes
    def predict_f(
            self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data
        points.
        """
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X, Y = self.data
        phi = self.kernel.compute_feature_vector(X)
        phi_new = self.kernel.compute_feature_vector(Xnew)
        sigma_2 = tf.squeeze(self.likelihood.variance_at(X), axis=-1)
        sigma = tf.math.sqrt(sigma_2)
        phi = tf.transpose(tf.transpose(phi) / sigma)
        # regularized_inverse_id = tf.linalg.tensor_diag(tf.math.reciprocal(sigma_2))
        inner_matrix = tf.eye(phi.shape[1], dtype=tf.float64) + tf.transpose(phi) @ phi
        L = tf.linalg.cholesky(inner_matrix)
        v = (Y - self.mean_function(X)) / sigma[..., None]
        Linv_phi = tf.linalg.triangular_solve(L, tf.transpose(phi), lower=True)
        # mat_inv = regularized_inverse_id - tf.transpose(Linv_phi) @ Linv_phi

        f_mean = phi_new @ (tf.transpose(phi)  @ v) \
                 - phi_new @ (tf.transpose(phi) @ (tf.transpose(Linv_phi) @ (Linv_phi @ v))) \
                 + self.mean_function(Xnew)
        # f_var = phi_new @ tf.transpose(phi_new) \
        #         - phi_new @ (tf.transpose(phi) @ (phi @ tf.transpose(phi_new)))\
        #         + phi_new @ (tf.transpose(phi) @ (tf.transpose(Linv_phi) @ (Linv_phi @ (phi @ tf.transpose(phi_new)))))
        f_var = tf.reduce_sum(tf.math.square(phi_new), axis=-1) \
               - tf.reduce_sum(tf.math.square(phi_new @ (tf.transpose(phi))), axis=-1) \
               + tf.reduce_sum(tf.math.square(phi_new @ (tf.transpose(phi) @ (tf.transpose(Linv_phi)))), axis=-1)
        return f_mean, tf.expand_dims(f_var, axis=-1)

class SquaredExponentialNystrom(gpflow.kernels.Kernel):
    @check_shapes(
        "exact_kernel.variance: [broadcast n_active_dims]",
    )
    def __init__(
            self, nystrom_centers: np.ndarray, exact_kernel: gpflow.kernels.RBF,
            active_dims: Optional[ActiveDims] = None
    ) -> None:
        """
        :param variance: the (initial) value for the variance parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the number of active dimensions e.g. [1., 1., 1.]
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(active_dims)
        self.exact_kernel = exact_kernel
        self.variance = exact_kernel.variance
        self.lengthscales = exact_kernel.lengthscales
        self.nystrom_centers = nystrom_centers
        self.n_centers = nystrom_centers.shape[0]
        self._validate_ard_active_dims(self.variance)
        self.Kmm_inv_sqrt = tf.linalg.sqrtm(tf.linalg.inv(self.exact_kernel.K(nystrom_centers)
                                                          + 1e-4 * tf.eye(nystrom_centers.shape[0], dtype=tf.float64)))

    def compute_feature_vector(self, X: TensorType) -> TensorType:
        return self.exact_kernel.K(X, self.nystrom_centers) @ self.Kmm_inv_sqrt

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        if X2 is None:
            return self.exact_kernel.K(X, X)
        else:
            return self.exact_kernel.K(X, X2)

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        return tf.linalg.diag_part(self.exact_kernel.K(X, X))

class NystromGPR(gpflow.models.GPR):
    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "noise_variance: []",
    )
    def __init__(self, data: RegressionData,
                 kernel: gpflow.kernels.SquaredExponential,
                 nystrom_centers = np.ndarray,
                 mean_function: Optional[MeanFunction] = None,
                 noise_variance: Optional[TensorData] = None,
                 likelihood: Optional[Gaussian] = None, ):
        super().__init__(data, kernel, mean_function, noise_variance, likelihood)
        self.nystrom_centers = nystrom_centers
        self.Kmm = self.kernel.K(nystrom_centers)
        self.Kmm_inv = tf.linalg.inv(self.Kmm + 1e-4 * tf.eye(self.Kmm.shape[0], dtype=tf.float64))
        self.Kmn = self.kernel.K(nystrom_centers, data[0])

    @check_shapes(
        "return: []",
    )
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        phi = self.kernel.compute_feature_vector(X)
        sigma_2 = tf.squeeze(self.likelihood.variance_at(X))
        regularized_inverse_id = tf.linalg.tensor_diag(tf.math.reciprocal(sigma_2))
        inner_mat = tf.eye(phi.shape[1], dtype=tf.float64) + tf.transpose(phi) @ regularized_inverse_id @ phi
        L = tf.linalg.cholesky(inner_mat)

        Lphi_inv = tf.linalg.triangular_solve(L, tf.transpose(phi) @ regularized_inverse_id, lower=True)
        mat_inv = regularized_inverse_id - tf.transpose(Lphi_inv) @ Lphi_inv  # GP is homoscedastic
        v = Y - self.mean_function(X)
        log_prob = - 0.5 * tf.transpose(v) @ mat_inv @ v \
                   - 0.5 * (2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L))) + tf.reduce_sum(
            tf.math.log(sigma_2))) \
                   - 0.5 * Y.shape[0] * tf.math.log(2 * np.float64(np.pi))
        return tf.reduce_sum(log_prob)

    @inherit_check_shapes
    def predict_f(
            self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data
        points.
        """
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X, Y = self.data
        k_new_m = self.kernel.K(Xnew, self.nystrom_centers)
        sigma_2 = tf.squeeze(self.likelihood.variance_at(X))
        regularized_inverse_id = tf.linalg.tensor_diag(tf.math.reciprocal(sigma_2))
        inner_matrix = self.Kmm + self.Kmn @ regularized_inverse_id @ tf.transpose(self.Kmn)
        L = tf.linalg.cholesky(inner_matrix + 1e-4 * tf.eye(tf.shape(inner_matrix)[0], dtype=tf.float64))
        Linv_phi = tf.linalg.triangular_solve(L, self.Kmn @ regularized_inverse_id, lower=True)
        mat_inv = regularized_inverse_id - tf.transpose(Linv_phi) @ Linv_phi

        f_mean = k_new_m @ (self.Kmm_inv @ (self.Kmn @ (mat_inv @ Y))) + self.mean_function(Xnew)
        f_var = k_new_m @ self.Kmm_inv @ tf.transpose(k_new_m) - k_new_m @ (
                    self.Kmm_inv @ self.Kmn @ (mat_inv @ tf.transpose(k_new_m @ self.Kmm_inv @ self.Kmn)))
        return f_mean, tf.expand_dims(tf.linalg.diag_part(f_var), axis=-1)
