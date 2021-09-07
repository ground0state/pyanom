"""
Copyright (c) 2019-2021 ground0state. All rights reserved.
License: MIT License
"""
import sys
import warnings

import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pyanom.utils import zscore, check_input_shape


FLOAT_MAX = sys.float_info.max
FLOAT_MIN = sys.float_info.min


class GraphicalLasso(BaseEstimator, DensityMixin):
    """Graphical lasso.

    Parameters
    ----------
    rho: float
        Inverse of the scale. The larger this is,
        precision matrix elements become sparse.
    max_iter: int
        Max iteration.
    max_iter_beta: int
        Max iteration of graphical lasso.
    tol: float
        When the update amount of the precision matrix
        becomes smaller than this tolerance value,
        the coordinate descent stops.
    tol_beta: float
        When the update amount of beta
        becomes smaller than this tolerance,
        the graphical lasso stops.
    """

    def __init__(self, rho=0.01, max_iter=100, max_iter_beta=1000,
                 tol=1e-8, tol_beta=1e-8):
        self.rho = rho
        self.max_iter = max_iter
        self.max_iter_beta = max_iter_beta
        self.tol = tol
        self.tol_beta = tol_beta

        self.cov_ = None
        self.pmatrix_ = None
        self.pmatrix_inv_ = None
        self.n_features_ = None

    def _initialize(self, X):
        self.n_features_ = X.shape[1]

    def _check_params(self, X):
        if self.n_features_ <= 1:
            raise ValueError("Feature size must be >=2")

    def _check_test_data(self, X):
        if X.shape[1] != self.n_features_:
            raise ValueError("Feature size must be same as training data")

    def fit(self, X):
        """Fit the model according to the given train data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Normal measured vectors,
            where n_samples is the number of samples.
        Returns
        -------
        self : object
        """
        # validation
        X = check_array(X)
        self._initialize(X)
        self._check_params(X)
        # normalize
        X = zscore(X, axis=0)

        # solve precision matrix
        self.pmatrix_, self.pmatrix_inv_, self.cov_ \
            = self._solve(X, rho=self.rho,
                          max_iter=self.max_iter,
                          max_iter_beta=self.max_iter_beta,
                          tol=self.tol,
                          tol_beta=self.tol_beta)
        return self

    def _solve(self, X, rho,
               max_iter, max_iter_beta,
               tol=None, tol_beta=None):
        # covariance
        cov = np.cov(X, rowvar=False, bias=False)

        # initialize precision matrix and inverse of precision matrix
        pmatrix = np.ones((X.shape[1], X.shape[1]))
        pmatrix_inv = cov + rho * np.diag(np.ones(X.shape[1]))

        # Coordinate Descent
        pmatrix, pmatrix_inv = self._coordinate_descent(
            cov, pmatrix, pmatrix_inv, rho,
            max_iter, max_iter_beta, tol, tol_beta)

        return pmatrix, pmatrix_inv, cov

    def _coordinate_descent(self, cov, pmatrix, pmatrix_inv, rho,
                            max_iter, max_iter_beta, tol, tol_beta):

        pmatrix = pmatrix.copy()
        pmatrix_inv = pmatrix_inv.copy()
        pmatrix_inv_prev = np.copy(pmatrix_inv)
        for _ in range(max_iter):
            for i in range(len(cov)):
                W = np.delete(np.delete(pmatrix_inv, i, 0), i, 1)
                s = np.delete(cov[:, i], i, axis=0)

                # graphical lasso
                beta = self._glasso(W, s, rho,
                                    max_iter_beta, tol_beta)

                # update pmatrix_inv
                w = beta @ W
                sigma = cov[i, i] + rho
                w_ = np.insert(w, i, sigma)
                pmatrix_inv[:, i] = np.copy(w_)
                pmatrix_inv[i, :] = np.copy(w_)

                # update pmatrix
                lam = 1 / (sigma - beta @ W @ beta)
                l = - lam * beta
                l_ = np.insert(l, i, lam)
                pmatrix[:, i] = np.copy(l_)
                pmatrix[i, :] = np.copy(l_)

            if np.all(np.abs(
                    pmatrix_inv_prev - pmatrix_inv) <= tol):
                return pmatrix, pmatrix_inv

            pmatrix_inv_prev = pmatrix_inv

        warnings.warn('Coordinate descent not converged.', UserWarning)
        return pmatrix, pmatrix_inv

    def _glasso(self, W, s, rho, max_iter_beta, tol_beta, beta_init_coef=0.1):
        W_offdiag = W - np.diagflat(np.diag(W))
        beta = np.ones(W.shape[0]) * beta_init_coef
        for _ in range(max_iter_beta):
            beta_prev = beta.copy()
            A = s - beta @ W_offdiag
            for idx, a in enumerate(A):
                beta[idx] = np.sign(
                    a) * np.maximum(np.abs(a) - rho, 0.0) / W[idx, idx]
            if np.all(np.abs(beta - beta_prev) <= tol_beta):
                return beta
        warnings.warn('Glasso not converged.', UserWarning)
        return beta

    def score(self, X):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples, n_features)
            Anomaly score.
        """
        # validation
        check_is_fitted(self)
        X = check_array(X)
        self._check_test_data(X)
        # normalize
        X = zscore(X, axis=0)

        # calculate anomaly score
        diag = np.diag(self.pmatrix_)
        anomaly_score = []
        for x in X:
            a = np.log(2 * np.pi / diag) / 2 \
                + (x @ self.pmatrix_)**2 / (2 * diag)
            anomaly_score.append(a)

        return np.array(anomaly_score)

    @staticmethod
    def anomaly_analysis_score(pmatrix1, cov1, pmatrix2, cov2):
        """Calculate anomaly score for each feature according to the given test data.

        Parameters
        ----------
        pmatrix1 : array-like, shape (n_features, n_features)
            Precision matrix of normal measured vectors.

        cov1: array-like, shape (n_features, n_features)
            Covariance matrix of normal measured vectors.

        pmatrix2 : array-like, shape (n_features, n_features)
            Precision matrix of error measured vectors.

        cov2: array-like, shape (n_features, n_features)
            Covariance matrix of error measured vectors.

        Returns
        -------
        anomaly_score : ndarray, shape (n_features, )
            Anomaly score.
        """
        pmatrix1_diag = np.diag(pmatrix1)
        pmatrix2_diag = np.diag(pmatrix2)
        a = 1 / 2 * np.log(pmatrix1_diag / pmatrix2_diag) - 1 / \
            2 * (np.diag(pmatrix1 @ cov1 @ pmatrix1) / pmatrix1_diag -
                 np.diag(pmatrix2 @ cov1 @ pmatrix2) / pmatrix2_diag)
        b = 1 / 2 * np.log(pmatrix2_diag / pmatrix1_diag) - 1 / \
            2 * (np.diag(pmatrix2 @ cov2 @ pmatrix2) / pmatrix2_diag -
                 np.diag(pmatrix1 @ cov2 @ pmatrix1) / pmatrix1_diag)
        return np.maximum(a, b)


class DirectLearningSparseChanges(BaseEstimator):
    """Direct Learning of Sparse Changes
    in Markov Networks by Density Ratio Estimation.

    Parameters
    ----------
    lambda1 : float
        L2 penalty.
    lambda2 : float
        L1 penalty.
    lr: float
        Learning rate.
    max_iter: int
        Max number of iterations over the train dataset
        to perform training.
    tol: float
        Tolerance for termination.
        A lower bound on the change in the value
        of the objective function during a step.
    """

    def __init__(self, lambda1, lambda2,
                 lr=0.01, max_iter=1000,
                 tol=1e-4):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self._loss = []

        self.theta_ = None
        self.S_ = None
        self.G_ = None

    def fit(self, X_normal, X_error):
        """Fit the DirectLearningSparseChanges model
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
        """
        # validation
        X_normal = check_array(X_normal)
        X_error = check_array(X_error)
        check_input_shape(X_normal, X_error)
        # normalize
        X_normal = zscore(X_normal, axis=0)
        X_error = zscore(X_error, axis=0)

        # initialize
        self.S_ = np.cov(X_normal, rowvar=False)
        self.G_ = -1 / 2 * np.asarray([np.outer(x, x) for x in X_error]).T

        # initialize theta
        self.theta_ = self.S_

        self._loss = []
        immediate_loss_prev = FLOAT_MAX
        for _ in range(self.max_iter):
            self.theta_ = self._prox(self.theta_ - self.lr *
                                     self._obj_deri_func(self.theta_),
                                     self.lr, self.lambda2)

            immediate_loss = self._obj_func(self.theta_)
            self._loss.append(immediate_loss)
            if immediate_loss_prev - immediate_loss <= self.tol:
                break
            immediate_loss_prev = immediate_loss
        else:
            warnings.warn(
                "Objective did notconverge. The max_iter was reached.")
        return self

    def _prox(self, v, eta, lam):
        return np.sign(v) * np.maximum(np.abs(v) - eta * lam, 0.0)

    def get_sparse_changes(self):
        """Gettter for sparse changes.
        Returns
        -------
        theta : The difference of precision matrix.
        """
        check_is_fitted(self)
        return self.theta_

    def _obj_func(self, theta):
        temp = np.zeros(self.G_.shape[2])
        for i in range(self.G_.shape[0]):
            for j in range(self.G_.shape[1]):
                temp = self.G_[i, j, :] * theta[j, i]

        obj = 1 / 2 * np.trace(theta @ self.S_) \
            + np.log(np.mean(np.exp(temp))) \
            + 1 / 2 * self.lambda1 * np.sum(theta**2) \
            + self.lambda2 * np.sum(np.abs(theta))
        return obj

    def _obj_deri_func(self, theta):
        dobj = self.S_ / 2

        temp = np.zeros(self.G_.shape[2])
        for i in range(self.G_.shape[0]):
            for j in range(self.G_.shape[1]):
                temp = self.G_[i, j, :] * theta[j, i]

        dobj += self.G_ @ np.exp(temp) / np.sum(np.exp(temp))
        dobj += self.lambda1 * theta

        return dobj


class DirectLearningSparseChangesDual(BaseEstimator):
    """Direct Learning of Sparse Changes
    in Markov Networks by Density Ratio Estimation.
    Parameters
    ----------
    lambda1 : float
        L2 penalty.
    lambda2 : float
        Additional L2 penalty.
    lr: float
        Learning rate.
    max_iter: int
        Max number of iterations over the train dataset
        to perform training.
    tol: float
        Tolerance for termination.
        A lower bound on the change in the value
        of the objective function during a step.
    """

    def __init__(self, lambda1, lambda2, lr=1e-8,
                 max_iter=1000, tol=1e-3):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self._loss = []

        self.pmatrix_diff_ = None
        self.G_ = None
        self.H_ = None
        self.alpha_ = None

    def fit(self, X_normal, X_error):
        """Fit the DirectLearningSparseChanges model
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
        """
        # validation
        X_normal = check_array(X_normal)
        X_error = check_array(X_error)
        check_input_shape(X_normal, X_error)
        # normalize
        X_normal = zscore(X_normal, axis=0)
        X_error = zscore(X_error, axis=0)

        # initialize
        self.G_ = -1 / 2 * np.cov(X_normal, rowvar=False)
        self.H_ = -1 / 2 * np.asarray([np.outer(x, x) for x in X_error]).T

        # initialize alpha
        temp = np.ones(X_error.shape[0])
        self.alpha_ = temp / temp.sum()

        self._loss = []
        immediate_loss_prev = FLOAT_MAX
        for i in range(self.max_iter):
            t = 1 / (i + 1)
            temp = self.alpha_ \
                - self.lr * (self._obj_deri_func(self.alpha_)
                             - t / np.maximum(self.alpha_,
                                              FLOAT_MIN)
                             + 2 * (np.sum(self.alpha_)
                                    - 1) / np.sqrt(t))
            self.alpha_ = temp

            immediate_loss = self._obj_func(
                self.alpha_) - t * np.sum(np.log(self.alpha_)) \
                + (np.sum(self.alpha_) - 1)**2 / np.sqrt(t)
            self._loss.append(immediate_loss)
            if immediate_loss_prev - immediate_loss <= self.tol:
                break
            immediate_loss_prev = immediate_loss
        else:
            warnings.warn(
                "Objective did notconverge. The max_iter was reached.")

        theta = self._trans_dual(self.alpha_)
        self.pmatrix_diff_ = self._theta2pmatrix(theta)

        return self

    def get_sparse_changes(self):
        """Gettter for sparse changes.
        Returns
        -------
        pmatrix_diff : The difference of precision matrix.
        """
        check_is_fitted(self)
        return self.pmatrix_diff_

    def _obj_func(self, alpha):
        xi = self.G_ - self.H_ @ alpha

        obj = np.sum(sp.special.xlogy(alpha, alpha))
        temp = (np.maximum(np.abs(xi) - self.lambda2, 0))**2
        temp = np.triu(temp)
        temp = np.sum(temp)
        obj += 1 / (2 * self.lambda1) * temp
        return obj

    def _obj_deri_func(self, alpha):
        xi = self.G_ - self.H_ @ alpha

        # calculate gamma
        gamma = np.zeros(xi.shape)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                if xi[i, j] > self.lambda2:
                    gamma[i, j] = -(xi[i, j] - self.lambda2) / self.lambda1
                elif xi[i, j] < -self.lambda2:
                    gamma[i, j] = -(xi[i, j] + self.lambda2) / self.lambda1
                else:
                    gamma[i, j] = 0

        # calculate derivative
        dobj = np.log(
            np.maximum(alpha, sys.float_info.min)) + 1

        temp = np.zeros(self.H_.shape[2])
        for i in range(gamma.shape[0]):
            for j in range(gamma.shape[1]):
                if j < i:
                    continue
                temp += gamma[i, j] * self.H_[i, j, :]
        dobj += temp
        return dobj

    def _trans_dual(self, alpha):
        xi = self.G_ - self.H_ @ alpha

        theta = np.zeros(xi.shape)
        it = np.nditer(xi, flags=['multi_index'])
        while not it.finished:
            norm = np.abs(it[0])
            if norm > self.lambda2:
                theta[it.multi_index] = 1 / self.lambda1 * \
                    (1 - self.lambda2 / norm) * it[0]
            else:
                theta[it.multi_index] = 0

            it.iternext()

        return theta

    def _theta2pmatrix(self, theta):
        diag = np.diag(np.diag(theta))
        return (theta + diag) / 2
