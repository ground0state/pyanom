"""
Copyright (c) 2019-2021 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from pyanom.utils import check_array_feature_dimension, check_array_type


class SSA(BaseEstimator):
    """Singular spectrum analysis for anomaly detection.

    Parameters
    ----------
    window_size; int
        Sliding window size for making partial time series.

    trajectory_n: int
        Number of row of trajectory matrix.

    trajectory_pattern: int
        Number of trajectory matrix's left singular vectors selected as principal subspace.

    test_n: int
        Number of row of test matrix.

    test_pattern: int
        Number of test matrix's left singular vectors selected as principal subspace.

    lag: int
        Lag between trajectory matrix and test matrix.

    Returns
    -------
    self : object

    """

    def __init__(self,
                 window_size=50,
                 trajectory_n=25,
                 trajectory_pattern=3,
                 test_n=25,
                 test_pattern=2,
                 lag=25):
        self.window_size = window_size
        self.trajectory_n = trajectory_n
        self.trajectory_pattern = trajectory_pattern
        self.test_n = test_n
        self.test_pattern = test_pattern
        self.lag = lag

        self.score_ = None

    def _check_params(self, y):
        assert self.window_size < len(y) + 1
        assert self.trajectory_pattern <= self.window_size
        assert self.test_pattern <= self.window_size
        assert self.trajectory_n >= 1
        assert self.test_n >= 1
        assert 0 <= self.lag < len(y) - self.window_size - self.test_n - 1

    def fit(
        self,
        y,
    ):
        """Fit the DensityRatioEstimation model according to the given data.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            measured vectors contain error, where n_samples is the number of samples.

        Returns
        -------
        self : object
        """

        # validation
        y = check_array_type(y)
        check_array_feature_dimension(y, 1)
        y = y.reshape(-1)
        self._check_params(y)

        X = np.asarray([y[i:i + self.window_size]
                        for i in range(len(y) - self.window_size - 1)])

        anomaly_score = []
        for t in range(self.window_size + self.test_n + 1, len(y) - self.lag):
            # trajectory matrix and test matrix at t
            X_t = X[t - self.trajectory_n - self.window_size:t - self.window_size].T
            Z_t = X[t - self.test_n + self.lag -
                    self.window_size:t - self.window_size + self.lag].T

            # SVD
            U, s, _ = np.linalg.svd(X_t)
            U = U[:, :self.trajectory_pattern]

            Q, _, _ = np.linalg.svd(Z_t)
            Q = Q[:, :self.test_pattern]

            UhQ = np.dot(U.T, Q)
            _, s, _ = np.linalg.svd(UhQ)

            a = 1 - s[0]
            # regularize
            if a < 10e-10:
                a = 0
            anomaly_score.append(a)

        self.score_ = np.array(anomaly_score)
        return self

    def score(self):
        """Calculate anomaly score for each feature according to the given data.

         Returns
         -------
         score : ndarray, shape (n_samples,)
            Anomaly score.
         """
        check_is_fitted(self)
        return self.score_
