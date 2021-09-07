"""
Copyright (c) 2019-2021 ground0state. All rights reserved.
License: MIT License
"""
import sys
import warnings

import numpy as np
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pyanom.utils import check_input_shape

FLOAT_MAX = sys.float_info.max


class KLDensityRatioEstimator(BaseEstimator, DensityMixin):
    """Kullback-Leibler density ratio estimation.

    Parameters
    ----------
    band_width : float
        Smoothing parameter gaussian kernel.
    lr: float
        Learning rate.
    max_iter: int
        Max number of iterations over the train dataset
        to perform training.
    tol: float
        When the update object function
        becomes smaller than this tolerance value,
        update stops.
    """

    def __init__(self, band_width=1.0, lr=0.1, max_iter=100, tol=1e-4):
        self.band_width = band_width
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter

        self.theta_ = None
        self.X_center_ = None
        self.psi_ = None
        self.psi_prime_ = None
        # losses of objective function in training
        self.loss_ = []

    def fit(self, X_normal, X_error):
        """Fit the DensityRatioEstimation model
        according to the given train data.
        Parameters
        ----------
        X_normal : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.
        X_error: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
        Notes
        -----
        Use X_normal for basic function.
        """
        # validation
        X_normal = check_array(X_normal)
        X_error = check_array(X_error)
        check_input_shape(X_normal, X_error)

        # prepare basic function
        self.X_center_ = X_normal
        self.psi_ = np.asarray([self._gaussian_kernel(x, X_normal)
                                for x in X_normal])
        self.psi_prime_ = np.asarray(
            [self._gaussian_kernel(x, X_normal) for x in X_error])

        # initialize theta
        self.theta_ = np.ones(len(self.psi_)) / len(self.psi_)

        # initilalize density latio
        r = np.dot(self.psi_, self.theta_)
        r_prime = np.dot(self.psi_prime_, self.theta_)

        # execute gradient method
        self.loss_ = []
        immediate_loss_prev = FLOAT_MAX

        for _ in range(self.max_iter):
            # update theta
            temp = self.theta_ - self.lr * self._obj_deri_func(r)
            self.theta_ = np.maximum(temp, 0)

            # calculate density latio
            r = np.dot(self.psi_, self.theta_)
            r_prime = np.dot(self.psi_prime_, self.theta_)

            # calculate loss
            immediate_loss = self._obj_func(r, r_prime)
            self.loss_.append(immediate_loss)

            if immediate_loss_prev - immediate_loss <= self.tol:
                break
            immediate_loss_prev = immediate_loss
        else:
            warnings.warn(
                "Objective did notconverge. The max_iter was reached.")

        return self

    def _obj_func(self, r, r_prime):
        obj = np.sum(r_prime) / len(self.psi_prime_) \
            - np.sum(np.log(r)) / len(self.psi_)
        return obj

    def _obj_deri_func(self, r):
        dobj = self.psi_prime_.sum(axis=0) / len(self.psi_prime_) - \
            (self.psi_ / r).sum(axis=0) / len(self.psi_)
        return dobj

    def _gaussian_kernel(self, x, X):
        psi = np.exp(-np.sum((x - X)**2, axis=1) / (2 * self.band_width**2))
        return psi

    def oof_score(self, X_normal, X_error):
        """Calculate objective according to the given oof data.
        Parameters
        ----------
        X_normal : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.
        X_error: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.
        Returns
        -------
        obj : array-like, shape (n_samples,)
            Objective.
        """
        # validation
        check_is_fitted(self)
        X_normal = check_array(X_normal)
        X_error = check_array(X_error)
        check_input_shape(X_normal, X_error)

        psi = np.asarray([self._gaussian_kernel(x, self.X_center_)
                          for x in X_normal])
        psi_prime = np.asarray(
            [self._gaussian_kernel(x, self.X_center_) for x in X_error])

        r = np.dot(psi, self.theta_)
        r_prime = np.dot(psi_prime, self.theta_)

        obj = np.sum(r_prime) / len(psi_prime) \
            - np.sum(np.log(r)) / len(psi)
        return obj

    def score(self, X_error):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X_error: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """
        # validation
        check_is_fitted(self)
        X_error = check_array(X_error)
        check_input_shape(self.X_center_, X_error)

        psi_prime = np.asarray([self._gaussian_kernel(x, self.X_center_)
                                for x in X_error])
        r_prime = np.dot(psi_prime, self.theta_)
        anomaly_score = -np.log(r_prime)
        return anomaly_score
