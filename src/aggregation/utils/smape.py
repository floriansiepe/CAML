import numpy as np

EPSILON = 1e-10
def smape(A, F):
    return np.mean(2.0 * np.abs(A - F) / ((np.abs(A) + np.abs(F)) + EPSILON))
