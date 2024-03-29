import unittest

import numpy as np


class TestGraphicalLasso(unittest.TestCase):
    """Basic test cases."""

    def _getTarget(self):
        from pyanom.structure_learning import GraphicalLasso
        return GraphicalLasso

    def _makeOne(self, *args, **kwargs):
        return self._getTarget()(*args, **kwargs)

    @classmethod
    def setUpClass(self):
        self.X_normal = np.array([[0.975586009, -0.745997359, -0.229331244],
                                  [-0.460992487, -1.304668238, -0.599247488],
                                  [-0.503171745, -1.308368748, -1.451411048],
                                  [-0.904446243, -0.287837582, 0.197153592],
                                  [-1.106120624, 0.243612535, 1.051237763],
                                  [0.371920628, 1.690566027, -0.468645532],
                                  [-0.861682655, 1.472544046, -0.846863556],
                                  [0.632918214, 1.35895507, -1.217528827],
                                  [0.017011646, 1.556247275, -0.149119024],
                                  [-1.129336215, 0.486811944, 0.012272206],
                                  [0.498967152, -0.530065628, -2.14011938],
                                  [0.402460108, -0.474465633, -0.041584595],
                                  [-0.847994655, -1.281269721, -0.430338406],
                                  [-0.583857254, 0.228815073, -1.321443286],
                                  [0.963425438, -1.136873938, 0.990406269],
                                  [-1.342349795, -0.147133485, 1.286410605],
                                  [-0.546153552, 0.134343445, -0.380672316],
                                  [-2.264867999, 0.227795362, 1.477762968],
                                  [0.070095074, -0.770899782, 2.100831522],
                                  [0.425213005, 0.796156033, 1.676164975]])

        self.X_error = np.array([[-0.273095586, 0.356336588, 1.595876828],
                                 [-0.708547003, -0.572139833, 0.858932219],
                                 [-1.125947228, -1.049026454, 0.35980022],
                                 [0.653070988, -0.052417831, 0.787284547],
                                 [-1.059131881, 1.621161051, -1.295306533],
                                 [0.499065038, -1.064179225, 1.243325767],
                                 [0.452740621, -0.737171777, 0.352807563],
                                 [0.626897927, -1.100559392, -0.905560876],
                                 [1.338835274, 2.083549348, -1.280796042],
                                 [0.264928015, 10, 2.544472412],
                                 [-0.754827534, -1.031919195, 1.227285333],
                                 [-0.774019674, 0.241245625, -0.989132941],
                                 [1.298381426, 0.19445334, 2.267355363],
                                 [1.46892843, 1.24946146, 0.322341667],
                                 [1.057265661, -0.846614104, -0.355396321],
                                 [0.810670486, -0.719804484, -0.943762163],
                                 [1.169028226, 0.492444331, 0.234015505],
                                 [-0.307091024, -1.56195639, 0.509095939],
                                 [0.849156845, 0.533674261, 0.069183014],
                                 [0.102812565, 8, 1.545239732]])

    def test_outlier_analysis_score_shape(self):
        target = self._makeOne()
        target.fit(self.X_normal)
        pred = target.score(self.X_error)
        self.assertEqual(pred.shape, (20, 3))

    def test_incorrect_feature_size(self):
        X_normal = np.array([-0.056523959,
                             - 0.881470896,
                             -0.249935965,
                             0.186624902,
                             -0.30183287,
                             2.000815584,
                             0.710538188,
                             0.591089702,
                             0.099804538,
                             0.114730483]).reshape(-1, 1)

        X_error = np.array([0.660985506,
                            -1.450512173,
                            -1.27733756,
                            -1.420294211,
                            0.737179562,
                            1.481425898,
                            -0.170147132,
                            -1.527687346,
                            0.580282631,
                            -3.722489636]).reshape(-1, 1)
        target = self._makeOne()
        with self.assertRaises(ValueError):
            target.fit(X_normal)

    def test_anomaly_analysis_score_shape(self):
        target1 = self._makeOne()
        target1.fit(self.X_normal)
        target2 = self._makeOne()
        target2.fit(self.X_error)
        pred = self._getTarget().anomaly_analysis_score(target1.pmatrix_, target1.cov_, target2.pmatrix_, target2.cov_)
        self.assertEqual(pred.shape, (3, ))
        self.assertEqual(target1.pmatrix_.shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
