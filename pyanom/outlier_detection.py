"""
Copyright (c) 2019 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np
# from scipy.stats import f
from pyanom.utils import check_array_type, check_input_shape, check_array_feature_dimension, zscore, tensor_normalize


class CAD():
    """CUSUM Anomaly Detection.
    """

    def __init__(self):
        self.normal_mean = None
        self.normal_std = None
        self.error_mean = None
        self.nu = None
        self.uppper = None

    def fit(self, y, threshold):
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

        self.normal_mean = np.mean(y)
        self.normal_std = np.std(y)
        self.error_mean = threshold
        self.nu = self.error_mean - self.normal_mean

        if self.nu > 0:
            self.uppper = True
        else:
            self.uppper = False

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
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """
        # validation
        y_test = check_array_type(y_test)
        check_array_feature_dimension(y_test, 1)

        if self.uppper:
            anomaly_socre = self.nu * \
                (y_test - self.normal_mean - self.nu/2)/self.normal_std**2
        else:
            anomaly_socre = -1*self.nu * \
                (y_test - self.normal_mean + self.nu/2)/self.normal_std**2

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


class HotelingT2():
    """Hotelling's t-squared statistic.
    """

    def __init__(self):
        self.mean_val = None
        self.cov_val_inv = None
        self.M = None
        self.N = None

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

        self.N, self.M = X.shape
        self.mean_val = X.mean(axis=0)
        if self.M > 1:
            self.cov_val_inv = np.linalg.inv(np.cov(X, rowvar=0, bias=1))
        elif self.M == 1:
            self.cov_val_inv = np.array([1/np.var(X)])
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
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """
        # validation
        X = check_array_type(X)

        pred = []
        for x in X:
            if self.M > 1:
                a = (x-self.mean_val)@self.cov_val_inv@(x-self.mean_val)
            elif self.M == 1:
                a = (x-self.mean_val)**2*self.cov_val_inv

            # T2 = (self.N - self.M)/((self.N + 1) * self.M) * a
            # prob = f.pdf(T2, self.M, self.N-self.M)
            pred.append(a)

        return np.asarray(pred)


class DirectionalDataAnomalyDetection():
    def __init__(self):
        self.mean_val = None

    def fit(self, X, normalize=True):
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

        if normalize:
            X = tensor_normalize(X, axis=1)

        self.mean_val = X.mean(axis=0).reshape(-1, 1)

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
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """
        # validation
        X = check_array_type(X)

        anomaly_score = 1 - X@self.mean_val
        return anomaly_score
