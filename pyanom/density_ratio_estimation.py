"""
Copyright (c) 2019 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np
from pyanom.utils import check_array_type, check_input_shape


class KLDensityRatioEstimation():
    """Kullback-Leibler density ratio estimation.

    Parameters
    ----------
    band_width : float
        Smoothing parameter gaussian kernel.

    learning_rate: float
        Learning rate.

    num_iterations: int
        Number of iterations over the train dataset to perform training.
    """

    def __init__(self, band_width=1.0, learning_rate=0.1, num_iterations=100):
        self.band_width = band_width
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.Js = None
        self.psi = None
        self.psi_prime = None
        self.eps = 10e-15

    def fit(self, X_normal, X_error):
        """Fit the DensityRatioEstimation model according to the given train data.

        Parameters
        ----------
        X_normal : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        X_error: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object

        Notes
        -----
        Use X_normal for basic function.
        """
        # validation
        X_normal = check_array_type(X_normal)
        X_error = check_array_type(X_error)
        check_input_shape(X_normal, X_error)

        self.theta = np.ones(len(X_normal))
        self.Js = []
        self.psi = np.asarray([self._gaussian_kernel(x, X_normal)
                               for x in X_normal])
        self.psi_prime = np.asarray(
            [self._gaussian_kernel(x, X_normal) for x in X_error])
        dJ_1 = self.psi_prime.sum(axis=0) / len(X_error)

        for _ in range(self.num_iterations):
            # calculate J
            r = np.dot(self.psi, self.theta)
            r = np.maximum(r, self.eps)
            r_prime = np.dot(self.psi_prime, self.theta)
            r_prime = np.maximum(r_prime, self.eps)
            J = np.sum(r_prime)/len(X_error) - np.sum(np.log(r))/len(X_normal)
            self.Js.append(J)

            # calculate gradient
            dJ = dJ_1 - (self.psi / r).sum(axis=0) / len(X_normal)
            self.theta -= self.learning_rate * dJ
        self.Js = np.array(self.Js)

        return self

    def _gaussian_kernel(self, x, X):
        return np.exp(-np.sum((x - X)**2, axis=1)/(2*self.band_width**2))

    def get_running_loss(self):
        """Kullback-Leibler density ratio estimation.

        Returns
        -------
        Js : array-like, shape (num_iterations,)
            losses of objective function in training.
        """
        return self.Js

    def score(self, X_normal, X_error):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X_normal : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        X_error: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """
        # validation
        X_normal = check_array_type(X_normal)
        X_error = check_array_type(X_error)
        check_input_shape(X_normal, X_error)

        psi_prime = np.asarray([self._gaussian_kernel(x, X_normal)
                                for x in X_error])
        r_prime = np.dot(psi_prime, self.theta)
        r_prime = np.maximum(r_prime, self.eps)
        return -np.log(r_prime)
