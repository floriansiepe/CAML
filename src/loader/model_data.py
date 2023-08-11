from dataclasses import dataclass

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel


@dataclass
class ModelData:
    id: int
    model: GlobalForecastingModel
    test_x: TimeSeries
    test_y: TimeSeries
    train_x: TimeSeries
    train_y: TimeSeries
    validation_x: TimeSeries
    validation_y: TimeSeries
