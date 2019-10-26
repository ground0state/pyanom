# pyanom

![](https://img.shields.io/badge/python-3.6%7C3.7%7C3.8-green?style=plastic)
![](https://img.shields.io/badge/dynamic/json.svg?label=version&colorB=5f9ea0&query=$.version&uri=https://raw.githubusercontent.com/ground0state/pyanom/master/package.json&style=plastic)

This library is Python projects for anomaly detection. This contains these techniques.

- Kullback-Leibler desity estimation
- Singular spectrum analysis
- Graphical lasso
- CUSUM anomaly detection
- Hoteling T2
- Directional data anomaly detection

## REQUIREMENTS

- numpy
- pandas

## INSTALLATION

```bash
pip install pyanom
```

## USAGE

### Kullback-Leibler desity estimation

```python
from pyanom.density_ratio_estimation import KLDensityRatioEstimation

model = KLDensityRatioEstimation(
    band_width=0.1, learning_rate=0.1, num_iterations=100)
model.fit(X_normal, X_error)
anomaly_score = model.predict(X_normal, X_error)
```

### Singular spectrum analysis

```python
from pyanom.subspace_methods import SSA

model = SSA()
model.fit(y, window_size=50, trajectory_n=25, trajectory_pattern=3, test_n=25, test_pattern=2, lag=25)
anomaly_score = model.score()
```

### Graphical lasso

```python
from pyanom.structure_learning import GraphicalLasso

model = GraphicalLasso()
model.fit(X_normal, rho=0.01, normalize=True)
anomaly_score = model.outlier_analysis_score(X_error)
```

### CUSUM anomaly detection

```python
from pyanom.outlier_detection import CAD

model = CAD()
model.fit(y_normal, threshold=1)
anomaly_score = model.score(y_error)
```

### Hoteling T2

```python
from pyanom.outlier_detection import HotelingT2

model = HotelingT2()
model.fit(X_normal)
anomaly_score = model.score(X_error)
```

### Directional data anomaly DirectionalDataAnomalyDetection

```python
from pyanom.outlier_detection import HotelingT2

model = DirectionalDataAnomalyDetection()
model.fit(X_normal, normalize=True))
anomaly_score = model.score(X_error)
```
