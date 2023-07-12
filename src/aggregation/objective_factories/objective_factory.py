from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable, Any

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from optuna.trial import Trial


class ObjectiveFactory(ABC):
    @abstractmethod
    def create(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Union[TimeSeries, Sequence[TimeSeries]],
        validation_series: Union[TimeSeries, Sequence[TimeSeries]],
        validation_covariates: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Callable[[Trial], float]:
        raise NotImplementedError(
            "Subclasses should implement a trial objective."
        )

    @abstractmethod
    def build_model(
        self, params: dict[str, Any], **kwargs
    ) -> GlobalForecastingModel:
        raise NotImplementedError(
            "Subclasses should implement a model builder."
        )
