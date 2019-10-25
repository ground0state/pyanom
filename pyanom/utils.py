"""
Copyright (c) 2019 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np
import pandas as pd

import warnings

warnings.simplefilter(action='ignore')


def check_array_type(array):
    """Input validation on an array.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    if not isinstance(array, (list, tuple, np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError("Input is not array-like object")

    # convert
    array = np.array(array).astype('float64')

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if array.ndim >= 3:
        raise ValueError("Input ndim must be 2")

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Input is not number")

    if np.issubdtype(array.dtype, 'complex128'):
        raise ValueError("Input contains complex number")

    if np.isnan(array).any():
        raise ValueError("Input contains NaN")

    if np.isinf(array).any():
        raise ValueError("Input contains inf")

    return array


def check_input_shape(X1, X2):
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("Feature size is not same")
    return True
