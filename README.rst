pyanom
======

|image0| |image1|

This library is Python projects for anomaly detection. This contains
these techniques.

-  Kullback-Leibler desity estimation
-  Singular spectrum analysis
-  Graphical lasso
-  CUMSUM anomaly detection
-  Hoteling T2
-  Directional data anomaly detection

REQUIREMENTS
------------

-  numpy
-  pandas

INSTALLATION
------------

.. code:: bash

   pip install pyanom

USAGE
-----

Kullback-Leibler desity estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   from pyanom.density_ratio_estimation import KLDensityRatioEstimation

   X_normal = np.loadtxt("../input/normal_data.csv", delimiter=",")
   X_error = np.loadtxt("../input/error_data.csv", delimiter=",")

   model = KLDensityRatioEstimation(
       band_width=0.1, learning_rate=0.1, num_iterations=100)
   model.fit(X_normal, X_error)
   anomaly_score = model.predict(X_normal, X_error)

Singular spectrum analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   from pyanom.subspace_methods import SSA

   y_error = np.loadtxt("../input/timeseries_error2.csv", delimiter=",")

   model = SSA()
   model.fit(y_error, window_size=50, trajectory_n=25, trajectory_pattern=3, test_n=25, test_pattern=2, lag=25)
   anomaly_score = model.score()

Graphical lasso
~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   from pyanom.structure_learning import GraphicalLasso

   X_normal = np.loadtxt("../input/normal_data.csv", delimiter=",")
   X_error = np.loadtxt("../input/error_data.csv", delimiter=",")

   model = GraphicalLasso()
   model.fit(X_normal, rho=0.01, normalize=True)
   anomaly_score = model.outlier_analysis_score(X_error)

CUSUM anomaly detection
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   from pyanom.outlier_detection import CAD

   y_normal = np.loadtxt(
       "../input/timeseries_normal.csv", delimiter=",").reshape(-1, 1)
   y_error = np.loadtxt(
       "../input/timeseries_error.csv", delimiter=",").reshape(-1, 1)

   model = CAD()
   model.fit(y_normal, threshold=1)
   anomaly_score = model.score(y_error)

Hoteling T2
~~~~~~~~~~~

.. code:: python

   import numpy as np
   from pyanom.outlier_detection import HotelingT2

   X_normal = np.loadtxt("../input/normal_data.csv", delimiter=",")
   X_error = np.loadtxt("../input/error_data.csv", delimiter=",")

   model = HotelingT2()
   model.fit(X_normal)
   anomaly_score = model.score(X_error)

Directional data anomaly DirectionalDataAnomalyDetection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   from pyanom.outlier_detection import DirectionalDataAnomalyDetection

   X_normal = np.loadtxt(
       "../input/normal_direction_data.csv", delimiter=",")
   X_error = np.loadtxt("../input/error_direction_data.csv", delimiter=",")

   model = DirectionalDataAnomalyDetection()
   model.fit(X_normal, normalize=True))
   anomaly_score = model.score(X_error)

.. |image0| image:: https://img.shields.io/badge/python-3.6%7C3.7%7C3.8-green?style=plastic
.. |image1| image:: https://img.shields.io/badge/dynamic/json.svg?label=version&colorB=5f9ea0&query=$.version&uri=https://raw.githubusercontent.com/ground0state/pyanom/master/package.json&style=plastic

