import io
import unittest

import numpy as np


class TestSSA(unittest.TestCase):
    """Basic test cases."""

    def _getTarget(self):
        from pyanom.subspace_methods import SSA
        return SSA

    def _makeOne(self, *args, **kwargs):
        return self._getTarget()(*args, **kwargs)

    @classmethod
    def setUpClass(self):
        self.X_error = np.array([0.660985506,
                                 -1.450512173,
                                 -1.27733756,
                                 -1.420294211,
                                 0.737179562,
                                 1.481425898,
                                 -0.170147132,
                                 -1.527687346,
                                 0.580282631,
                                 -3.722489636,
                                 0.660985506,
                                 -1.450512173,
                                 -1.27733756,
                                 -1.420294211,
                                 0.737179562,
                                 1.481425898,
                                 -0.170147132,
                                 -1.527687346,
                                 0.580282631,
                                 -3.722489636])

    def test_score_shape(self):
        target = self._makeOne()
        window_size = 3
        trajectory_n = 2
        trajectory_pattern = 2
        test_n = 2
        test_pattern = 2
        lag = 3
        target.fit(self.X_error, window_size=window_size, trajectory_n=trajectory_n,
                   trajectory_pattern=trajectory_pattern, test_n=test_n, test_pattern=test_pattern, lag=lag)
        pred = target.score()
        self.assertEqual(pred.shape, (len(self.X_error)-2 *
                                      window_size-trajectory_n-test_n+1, ))
