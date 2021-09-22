# pyanom

![](https://img.shields.io/badge/python-3.6%7C3.7%7C3.8-green?style=plastic)
![](https://img.shields.io/badge/dynamic/json.svg?label=version&colorB=5f9ea0&query=$.version&uri=https://raw.githubusercontent.com/ground0state/pyanom/master/package.json&style=plastic)
[![Downloads](https://static.pepy.tech/personalized-badge/pyanom?period=total&units=none&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/pyanom)
[![Downloads](https://static.pepy.tech/personalized-badge/pyanom?period=month&units=none&left_color=grey&right_color=orange&left_text=Downloads/Month)](https://pepy.tech/project/pyanom)

This library is Python projects for anomaly detection. This contains these techniques.

- Kullback-Leibler desity estimation
- Singular spectrum analysis
- Graphical lasso
- CUMSUM anomaly detection
- Hoteling T2
- Directional data anomaly detection

## REQUIREMENTS

- numpy
- pandas
- scikit-learn
- scipy

## INSTALLATION

```bash
pip install pyanom
```

## USAGE

We have posted a usage example in the demo folder.

### Kullback-Leibler desity estimation

```python
import numpy as np
from pyanom.density_ratio_estimation import KLDensityRatioEstimator

X_normal = np.loadtxt("./data/normal_data.csv", delimiter=",")
X_error = np.loadtxt("./data/error_data.csv", delimiter=",")

model = KLDensityRatioEstimator(
    band_width=h, lr=0.001, max_iter=100000)
model.fit(X_normal, X_error)
anomaly_score = model.score(X_normal, X_error)
```

### Singular spectrum analysis

```python
import numpy as np
from pyanom.subspace_methods import SSA

y_error = np.loadtxt("./data/timeseries_error2.csv", delimiter=",")

model = SSA(window_size=50, trajectory_n=25, trajectory_pattern=3, test_n=25, test_pattern=2, lag=25)
model.fit(y_error)
anomaly_score = model.score()
```

### Graphical lasso

```python
import numpy as np
from pyanom.structure_learning import GraphicalLasso

X_normal = np.loadtxt("./data/normal_data.csv", delimiter=",")
X_error = np.loadtxt("./data/error_data.csv", delimiter=",")

model = GraphicalLasso(rho=0.1)
model.fit(X_normal)
anomaly_score = model.score(X_error)
```

### Direct learning sparse changes

```python
from pyanom.structure_learning import DirectLearningSparseChanges

model = DirectLearningSparseChanges(
    lambda1=0.1, lambda2=0.3, max_iter=10000)
model.fit(X_normal, X_error)
pmatrix_diff = model.get_sparse_changes()
```

### CUSUM anomaly detection

```python
import numpy as np
from pyanom.outlier_detection import CAD

y_normal = np.loadtxt(
    "./data/timeseries_normal.csv", delimiter=",").reshape(-1, 1)
y_error = np.loadtxt(
    "./data/timeseries_error.csv", delimiter=",").reshape(-1, 1)

model = CAD(threshold=1.0)
model.fit(y_normal)
anomaly_score = model.score(y_error)
```

### Hoteling T2

```python
import numpy as np
from pyanom.outlier_detection import HotelingT2

X_normal = np.loadtxt("./data/normal_data.csv", delimiter=",")
X_error = np.loadtxt("./data/error_data.csv", delimiter=",")

model = HotelingT2()
model.fit(X_normal)
anomaly_score = model.score(X_error)
```

### Anomaly detection of directional data

```python
import numpy as np
from pyanom.outlier_detection import AD3

X_normal = np.loadtxt(
    "./data/normal_direction_data.csv", delimiter=",")
X_error = np.loadtxt("./data/error_direction_data.csv", delimiter=",")

model = AD3()
model.fit(X_normal, normalize=True)
anomaly_score = model.score(X_error)
```

## Test

```bash
python -m unittest discover
```
