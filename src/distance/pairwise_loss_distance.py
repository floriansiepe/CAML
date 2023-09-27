from typing import Callable

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from src.loader.model_data import ModelData


class PairwiseLossDistance:
    def __init__(
        self,
        inner_loss_function: Callable[[TimeSeries, TimeSeries], float],
        predictor: Callable[
            [GlobalForecastingModel, int, TimeSeries, TimeSeries], TimeSeries
        ],
    ):
        self.inner_loss_function = inner_loss_function
        self.predictor = predictor

    def distance(self, x: ModelData, y: ModelData) -> float:
        series_x_on_y = y.validation_y.append(y.test_y)
        series_y_on_x = x.validation_y.append(x.test_y)
        covariate_x_on_y = y.validation_x.append(y.test_x)
        covariate_y_on_x = x.validation_x.append(x.test_x)
        loss_x_on_y = self._compute_loss(
            x.model, series_x_on_y, covariate_x_on_y, y.test_y
        )
        loss_y_on_x = self._compute_loss(
            y.model, series_y_on_x, covariate_y_on_x, x.test_y
        )
        dist = (loss_x_on_y + loss_y_on_x) / 2
        return dist

    def _compute_loss(
        self,
        model: GlobalForecastingModel,
        series: TimeSeries,
        covariates: TimeSeries,
        y: TimeSeries,
    ) -> float:
        prediction = self.predictor(model, len(y), series, covariates)
        return self.inner_loss_function(prediction, y)
