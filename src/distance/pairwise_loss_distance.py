from typing import Callable

import numpy as np
from sklearn.base import RegressorMixin

from src.loader.model_data import ModelData


class PairwiseLossDistance:
    def __init__(self, inner_loss_function: Callable[[np.ndarray, np.ndarray], float]):
        self.inner_loss_function = inner_loss_function

    def distance(self, x: ModelData, y: ModelData) -> float:
        loss_x_on_y = self._compute_loss(x.model, y.test_x.values, y.test_y.values)
        loss_y_on_x = self._compute_loss(y.model, x.test_x.values, x.test_y.values)
        return (loss_x_on_y + loss_y_on_x) / 2

    def _compute_loss(self, model: RegressorMixin, x: np.ndarray, y: np.ndarray) -> float:
        return self.inner_loss_function(model.predict(x), y)