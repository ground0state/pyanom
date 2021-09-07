"""
Copyright (c) 2019-2021 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_is_fitted

from pyanom.utils import (check_array_feature_dimension, check_array_type,
                          tensor_normalize)


class CAD(BaseEstimator):
    """CUSUM Anomaly Detection.

    Parameters
    ----------
    threshold: float
        Size of the shift that is to be detected.
    """

    def __init__(self, threshold):
        self.threshold = threshold

        self.normal_mean_ = None
        self.normal_std_ = None
        self.nu_ = None
        self.uppper_ = None

    def fit(self, y):
        """Fit the model according to the given train data.

        Parameters
        ----------
        y : array-like, shape (n_samples, )
            Normal measured vectors, where n_samples is the number of samples.

        threshold: float
            Size of the shift that is to be detected.

        Returns
        -------
        self : object
        """
        # validation
        y = check_array_type(y)
        check_array_feature_dimension(y, 1)

        self.normal_mean_ = np.mean(y)
        self.normal_std_ = np.std(y)
        error_mean_ = self.threshold
        self.nu_ = error_mean_ - self.normal_mean_

        if self.nu_ > 0:
            self.uppper_ = True
        else:
            self.uppper_ = False
        return self

    def score(self, y_test, cumsum_on=True):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        y_test : array-like, shape (n_samples,)
            Error measured vectors, where n_samples is the number of samples.

        cumsum_on: bool
            If True, return cumsumed anomaly score. If False, return pure anomaly score.

        Returns
        -------
        anomaly_score : ndarray, shape (n_samples,)
            Anomaly score.
        """
        # validation
        check_is_fitted(self)
        y_test = check_array_type(y_test)
        check_array_feature_dimension(y_test, 1)

        if self.uppper_:
            anomaly_socre = self.nu_ * \
                (y_test - self.normal_mean_ - self.nu_ / 2) / self.normal_std_**2
        else:
            anomaly_socre = -1 * self.nu_ * \
                (y_test - self.normal_mean_ + self.nu_ / 2) / self.normal_std_**2

        a_operated = 0
        anomaly_socre_cumsum = []
        for a in anomaly_socre:
            a += a_operated
            a_operated = np.maximum(a, 0)
            anomaly_socre_cumsum.append(a_operated)
        anomaly_socre_cumsum = np.array(anomaly_socre_cumsum)

        if cumsum_on:
            return anomaly_socre_cumsum
        else:
            return anomaly_socre


class HotelingT2(BaseEstimator, DensityMixin):
    """Hotelling's t-squared statistic.
    """

    def __init__(self):
        self.mean_val_ = None
        self.cov_val_inv_ = None
        self.M_ = None
        self.N_ = None

    def fit(self, X):
        """Fit the Hotelling's t-squared model according to the given train data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
        """
        # validation
        X = check_array_type(X)

        self.N_, self.M_ = X.shape
        self.mean_val_ = X.mean(axis=0)
        if self.M_ > 1:
            self.cov_val_inv_ = np.linalg.inv(np.cov(X, rowvar=0, bias=1))
        elif self.M_ == 1:
            self.cov_val_inv_ = np.array([1 / np.var(X)])
        else:
            raise ValueError("Input shape is incorrect")

        return self

    def score(self, X):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : ndarray, shape (n_samples,)
            Anomaly score.
        """
        # validation
        check_is_fitted(self)
        X = check_array_type(X)

        pred = []
        for x in X:
            if self.M_ > 1:
                a = (x - self.mean_val_) @ self.cov_val_inv_ @ (x - self.mean_val_)
            elif self.M_ == 1:
                a = (x - self.mean_val_)**2 * self.cov_val_inv_
            pred.append(a)
        return np.asarray(pred)


class AD3(BaseEstimator, DensityMixin):
    """Anomaly detection of directional data according to the von Mises-Fisher distribution.

    Parameters
    ----------
    normalize: bool
        If True, normalize input array.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

        self.mean_val_ = None

    def fit(self, X):
        """Fit the model according to the given train data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        normalize: bool
            If True, normalize input array.

        Returns
        -------
        self : object
        """
        # validation
        X = check_array_type(X)

        if self.normalize:
            X = tensor_normalize(X, axis=1)

        self.mean_val_ = X.mean(axis=0).reshape(-1, 1)

        return self

    def score(self, X):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : ndarray, shape (n_samples,)
            Anomaly score.
        """
        # validation
        check_is_fitted(self)
        X = check_array_type(X)

        anomaly_score = 1 - X @ self.mean_val_
        return anomaly_score
