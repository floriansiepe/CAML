from typing import Callable

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from src.loader.model_data import ModelData


class PairwiseLossDistance:
    def __init__(self, inner_loss_function: Callable[[TimeSeries, TimeSeries], float],
                 predictor: Callable[[GlobalForecastingModel, int, TimeSeries], TimeSeries]):
        self.inner_loss_function = inner_loss_function
        self.predictor = predictor

    def distance(self, x: ModelData, y: ModelData) -> float:
        loss_x_on_y = self._compute_loss(x.model, y.validation_x, y.test_y)
        loss_y_on_x = self._compute_loss(y.model, x.validation_x, x.test_y)
        return (loss_x_on_y + loss_y_on_x) / 2

    def _compute_loss(self, model: GlobalForecastingModel, x: TimeSeries, y: TimeSeries) -> float:
        prediction = self.predictor(model, len(y), x)
        return self.inner_loss_function(prediction, y)
