"""
Copyright (c) 2019 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np
import sys
from pyanom.utils import zscore, check_array_type


class GraphicalLasso():
    """Graphical lasso.
    """

    def __init__(self):
        self.cov = None
        self.pmatrix = None
        self.pmatrix_inv = None
        self.best_loss = None

        self.pmatrix_new = None
        self.pmatrix_inv_new = None
        self.cov_new = None
        self.best_loss_new = None

        self.feature_size = None

    def fit(self, X, rho=0.01, normalize=True):
        """Fit the model according to the given train data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples is the number of samples.

        rho: float
            Inverse of the scale. The smaller this is, precision matrix elements become sparse.

        normalize: bool
            If True, normalize input array.

        Returns
        -------
        self : object
        """
        # validation
        X = check_array_type(X)
        self.feature_size = X.shape[1]
        if self.feature_size <= 1:
            raise ValueError(f"Feature size must be >=2")

        self.pmatrix, self.pmatrix_inv, self.cov, self.best_loss = self._solve(
            X, rho=rho, normalize=normalize)

        return self

    def _solve(self, X, rho=0.01, normalize=True):
        """Caluculate precision matrix."""

        if normalize:
            X = zscore(X, axis=0)

        cov = np.cov(X, rowvar=0, bias=1)
        pmatrix = cov
        pmatrix_inv = cov
        best_loss = -sys.float_info.max
        stopping_count = 0
        while True:
            for i in range(X.shape[1]):
                W = np.delete(pmatrix_inv, i, 0)
                W = np.delete(W, i, 1)
                W_diagonal_zero = W - np.diagflat(np.diag(W))

                s = cov[:, i]
                s = np.delete(s, i, axis=0)

                # solve beta
                beta = np.ones(W.shape[0]) * 10
                best_d = sys.float_info.max
                stopping_count2 = 0
                while True:
                    A = s - beta@W_diagonal_zero
                    for idx, a in enumerate(A):
                        if a > rho:
                            beta[idx] = (a - rho)/W[idx, idx]
                        elif a < -rho:
                            beta[idx] = (a + rho)/W[idx, idx]
                        else:
                            beta[idx] = 0

                    target = beta@W - s + rho * np.sign(beta)
                    d = np.sum(target)

                    if d < best_d:
                        best_d = d
                        stopping_count2 = 0
                    else:
                        stopping_count2 += 1

                    if stopping_count2 >= 10:
                        break

                # update pmatrix_inv
                w = beta@W
                sigma = cov[i, i] + rho
                w_ = np.insert(w, i, sigma)
                pmatrix_inv[:, i] = w_
                pmatrix_inv[i, :] = w_

                # update pmatrix
                lam = 1 / (sigma - beta@W@beta)
                l = - lam * beta
                l_ = np.insert(l, i, lam)
                pmatrix[:, i] = l_
                pmatrix[i, :] = l_

            loss = -1*np.log(np.linalg.det(pmatrix)) + \
                np.trace(cov@pmatrix) - rho*np.sum(np.abs(pmatrix))

            if loss < best_loss:
                best_loss = loss
                stopping_count = 0
            else:
                stopping_count += 1

            if stopping_count >= 10:
                break

        return pmatrix, pmatrix_inv, cov, best_loss

    def get_precision_matrix():
        return self.self.pmatrix

    def outlier_analysis_score(self, X):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples, n_features)
            Anomaly score.
        """
        # validation
        X = check_array_type(X)
        if self.feature_size != X.shape[1]:
            raise ValueError(f"Feature size must be same as training data")

        diag = np.diag(self.pmatrix)
        anomaly_score = []
        for x in X:
            a = np.log(2*np.pi/diag)/2 + (x@self.pmatrix)**2/(2*diag)
            anomaly_score.append(a)
        return np.array(anomaly_score)

    def anomaly_analysis_score(self, X, rho=0.01, normalize=True):
        """Calculate anomaly score for each feature according to the given test data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        rho: float
            Inverse of the scale. The smaller this is, precision matrix elements become sparse.

        normalize: bool
            If True, normalize input array.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples, n_features)
            Anomaly score.

        precision_matrix : array-like, shape (n_features, n_features)
            Precision matrix of error measured vectors.
        """
        # validation
        X = check_array_type(X)
        if self.feature_size != X.shape[1]:
            raise ValueError(f"Feature size must be same as training data")

        self.pmatrix_new, self.pmatrix_inv_new, self.cov_new, self.best_loss_new = self._solve(
            X, rho=rho, normalize=normalize)

        diag = np.diag(self.pmatrix).reshape(-1)
        diag_new = np.diag(self.pmatrix_new).reshape(-1)
        diag_S = np.diag(self.pmatrix@self.cov@self.pmatrix).reshape(-1)
        diag_S_new = np.diag(self.pmatrix_new@self.cov_new@
                             self.pmatrix_new).reshape(-1)

        a = np.log(diag/diag_new)/2 - (diag_S/diag - diag_S_new/diag_new)/2

        return a, self.pmatrix_new
