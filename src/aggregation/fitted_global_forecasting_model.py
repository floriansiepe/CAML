from typing import Union, Optional, Sequence, Tuple, List, BinaryIO

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel, XGBModel, LightGBMModel, RegressionModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from src.aggregation.utils.io import load_pickle, dump_pickle


class FittedGlobalForecastingModel(GlobalForecastingModel):
    def __init__(self, model: GlobalForecastingModel):
        super().__init__()
        self.pipeline_series = None
        self.pipeline_future_covariates = None
        self.pipeline_past_covariates = None
        self.model = model

    def __repr__(self):
        return (
            f"FittedGlobalForecastingModel(model={self.model},"
            f"pipeline_series={self.pipeline_series},"
            f"pipeline_past_covariates={self.pipeline_past_covariates},"
            f"pipeline_future_covariates={self.pipeline_future_covariates}"
            f")"
        )

    @staticmethod
    def load_model(
            path: Union[str, BinaryIO], model_type
    ) -> "GlobalForecastingModel":
        cm = load_pickle(path)
        internal_model = None
        if model_type == "NBeats":
            internal_model = NBEATSModel.load(path + "_internal_model.pkl")
        elif model_type == "XGBoost":
            internal_model = XGBModel.load(path + "_internal_model.pkl")
        elif model_type == "LightGBM":
            internal_model = LightGBMModel.load(path + "_internal_model.pkl")
        elif model_type == "ExtraTreesRegressor":
            internal_model = RegressionModel.load(path + "_internal_model.pkl")
        cm.model = internal_model
        return cm

    def save(
            self, path: Optional[Union[str, BinaryIO]] = None, **pkl_kwargs
    ) -> None:
        dump_pickle(path, self)
        self.model.save(path + "_internal_model.pkl")

    def clever_fit(
            self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            val_covariates: Optional[
                Union[TimeSeries, Sequence[TimeSeries]]
            ] = None,
            **kwargs,
    ):
        if self.model.supports_future_covariates:
            self.fit(
                series=series,
                future_covariates=covariates,
                val_series=val_series,
                val_future_covariates=val_covariates,
                **kwargs,
            )
        else:
            self.fit(
                series=series,
                past_covariates=covariates,
                val_series=val_series,
                val_past_covariates=val_covariates,
                **kwargs,
            )
        return self

    def clever_predict(
            self,
            n: int,
            series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if self.model.supports_future_covariates:
            return self.predict(
                n=n, series=series, future_covariates=covariates
            )
        else:
            return self.predict(n=n, series=series, past_covariates=covariates)

    def fit(
            self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[
                Union[TimeSeries, Sequence[TimeSeries]]
            ] = None,
            future_covariates: Optional[
                Union[TimeSeries, Sequence[TimeSeries]]
            ] = None,
            **kwargs,
    ) -> "FittedGlobalForecastingModel":
        series_copy = series.copy()
        past_covariates_copy = (
            past_covariates.copy() if past_covariates is not None else None
        )
        future_covariates_copy = (
            future_covariates.copy() if future_covariates is not None else None
        )
        past_covariates_transformed = None
        future_covariates_transformed = None
        self.pipeline_series = Scaler(n_jobs=-1, verbose=True, global_fit=True)
        series_transformed = self.pipeline_series.fit_transform(series_copy)

        if past_covariates_copy is not None:
            self.pipeline_past_covariates = Scaler(
                n_jobs=-1, verbose=True, global_fit=True
            )
            past_covariates_transformed = (
                self.pipeline_past_covariates.fit_transform(
                    past_covariates_copy
                )
            )

        if future_covariates_copy is not None:
            self.pipeline_future_covariates = Scaler(
                n_jobs=-1, verbose=True, global_fit=True
            )
            future_covariates_transformed = (
                self.pipeline_future_covariates.fit_transform(
                    future_covariates_copy
                )
            )

        self.model.fit(
            series=series_transformed,
            future_covariates=future_covariates_transformed,
            past_covariates=past_covariates_transformed,
            # **kwargs,
        )
        return self

    def predict(
            self,
            n: int,
            series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            past_covariates: Optional[
                Union[TimeSeries, Sequence[TimeSeries]]
            ] = None,
            future_covariates: Optional[
                Union[TimeSeries, Sequence[TimeSeries]]
            ] = None,
            num_samples: int = 1,
            verbose: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        series_scaled = self.pipeline_series.transform(series)
        future_covariates_scaled = None
        past_covariates_scaled = None
        if future_covariates is not None:
            if self.pipeline_future_covariates is None:
                raise RuntimeError(
                    "The model was not fitted with future covariates"
                )
            future_covariates_scaled = (
                self.pipeline_future_covariates.transform(future_covariates)
            )
        if past_covariates is not None:
            if self.pipeline_past_covariates is None:
                raise RuntimeError(
                    "The model was not fitted with past covariates"
                )
            past_covariates_scaled = self.pipeline_past_covariates.transform(
                past_covariates
            )
        raw_preds = self.model.historical_forecasts(
            series=series_scaled,
            future_covariates=future_covariates_scaled,
            past_covariates=past_covariates_scaled,
            retrain=False,
            start=len(series) - n,
        )
        return self.pipeline_series.inverse_transform(raw_preds)

    @property
    def extreme_lags(
            self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        return self.model.extreme_lags

    @property
    def _model_encoder_settings(
            self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[List[int]],
        Optional[List[int]],
    ]:
        return self.model._model_encoder_settings
