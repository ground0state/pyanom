# pyanom Module Repository

![]（https://img.shields.io/badge/python-%7C3.6%7C3.7%7C3.8-green）

This library is Python projects for anomaly detection. This contains these techniques.

- Kullback-Leibler desity estimation

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
model.fit(train_normal_data, train_error_data)
anomaly_score = model.predict(normal_data, error_data)
```
