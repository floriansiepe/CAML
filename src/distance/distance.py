import logging

import numpy as np
from scipy.spatial.distance import pdist


def pdist_wrapper(x, metric, *, out=None, **kwargs):
    if type(x) is np.ndarray:
        return pdist(x, metric, out=out)
    else:
        return pdist_obj(x, metric, out=out, **kwargs)


def pdist_obj(x, metric, dtype=np.float, out=None, verbose=False, **kwargs):
    """
    Compute the distance between each pair of the input collection. This is a adaption of scipy.spatial.distance.pdist
    to work with objects instead of arrays.
    @param x: The input collection
    @param metric: The distance metric to use
    @param dtype: Data type of the returned matrix
    @param kwargs: Additional arguments to pass to the distance metric
    @return: The distance metric in vector form
    """
    n = len(x)
    out_size = (n * (n - 1)) // 2
    expected_shape = (out_size,)
    if out is None:
        dm = np.empty(expected_shape, dtype=dtype)
    else:
        if out.shape != expected_shape:
            raise ValueError("Output array has incorrect shape.")
        if out.dtype != dtype:
            raise ValueError("Output array has incorrect type.")
        dm = out
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if verbose and k % 1000 == 0:
                logging.info(f"Distance Metric Progress {k}/{out_size}")
            dm[k] = metric(x[i], x[j], **kwargs)
            k += 1
    return dm
