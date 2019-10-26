"""
Copyright (c) 2019 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np
from pyanom.utils import check_array_type, check_array_feature_dimension


class SSA():
    """Singular spectrum analysis for anomaly detection"""

    def __init__(self):
        self.__score = None

    def fit(self, y, window_size=50, trajectory_n=25, trajectory_pattern=3, test_n=25, test_pattern=2, lag=25):
        """Fit the DensityRatioEstimation model according to the given data.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            measured vectors contain error, where n_samples is the number of samples.

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
        assert window_size < len(y) + 1
        assert trajectory_pattern <= window_size
        assert test_pattern <= window_size
        assert trajectory_n >= 1
        assert test_n >= 1
        assert 0 <= lag < len(y) - window_size - test_n - 1

        # validation
        y = check_array_type(y)
        check_array_feature_dimension(y, 1)
        y = y.reshape(-1)

        X = np.asarray([y[i:i+window_size]
                        for i in range(len(y) - window_size - 1)])

        anomaly_score = []
        for t in range(window_size+test_n+1, len(y) - lag):
            # trajectory matrix and test matrix at t
            X_t = X[t-trajectory_n-window_size:t-window_size].T
            Z_t = X[t-test_n+lag -
                    window_size:t-window_size+lag].T

            # SVD
            U, s, _ = np.linalg.svd(X_t)
            U = U[:, :trajectory_pattern]

            Q, _, _ = np.linalg.svd(Z_t)
            Q = Q[:, :test_pattern]

            UhQ = np.dot(U.T, Q)
            _, s, _ = np.linalg.svd(UhQ)

            a = 1 - s[0]
            # regularize
            if a < 10e-10:
                a = 0
            anomaly_score.append(a)

        self.__score = np.array(anomaly_score)

        return self

    def score(self):
        """Calculate anomaly score for each feature according to the given data.

         Returns
         -------
         score : array-like, shape (n_samples,)
            Anomaly score.
         """
        return self.__score
