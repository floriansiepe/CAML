from typing import Tuple, Callable

import numpy as np
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.aggregation.fitted_global_forecasting_model import FittedGlobalForecastingModel
from src.aggregation.utils.smape import smape
from src.loader.model_data import ModelData


class TimeseriesEvaluator:
    def __init__(
        self,
        model_row,
        model: FittedGlobalForecastingModel,
        timeseries: dict[int, ModelData],
        predictor: Callable[
            [GlobalForecastingModel, int, TimeSeries, TimeSeries], TimeSeries
        ],
    ):
        self.model_row = model_row
        self.model = model
        self.timeseries = timeseries
        self.predictor = predictor

    def evaluate(
        self, model_row_validation_set=None
    ) -> Tuple[float, float, float, float, float]:
        if model_row_validation_set is None:
            model_row_validation_set = self.model_row
        model_id = model_row_validation_set["model_data"].id
        split_timeseries = self.timeseries[model_id]

        y_test = split_timeseries.test_y.pd_dataframe().values
        model_pred = self._evaluate_global_model(split_timeseries)

        assert len(model_pred) == len(y_test), (
            "Length of prediction and test set must be equal. Was: "
            f"{len(model_pred)} and {len(y_test)}"
        )

        model_mse = mean_squared_error(y_test, model_pred)
        model_mae = mean_absolute_error(y_test, model_pred)
        model_rmse = np.sqrt(model_mse)
        model_uncertainty = np.abs(model_mae)
        model_smape = smape(y_test, model_pred)

        return (
            model_mae,
            model_mse,
            model_rmse,
            model_uncertainty,
            model_smape,
        )

    def _evaluate_global_model(self, split_timeseries: ModelData):
        series = split_timeseries.train_y.append(split_timeseries.validation_y).append(
            split_timeseries.test_y
        )
        covariates = split_timeseries.train_x.append(
            split_timeseries.validation_x
        ).append(split_timeseries.test_x)

        preds = (
            self.predictor(
                self.model,
                len(split_timeseries.test_y),
                series,
                covariates,
            )
            .pd_dataframe()
            .values
        )
        return preds
