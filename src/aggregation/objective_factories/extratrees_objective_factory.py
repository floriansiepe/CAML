from typing import Union, Sequence, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import smape, mae
from darts.models import RegressionModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from optuna import Trial
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.aggregation.objective_factories.objective_factory import ObjectiveFactory

# log_10(x+1)
def log10p(x):
    return np.log10(1 + x)


# 10^x - 1
def exp10p(x):
    return 10**x - 1

class ExtraTreesObjectiveFactory(ObjectiveFactory):
    def create(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Union[TimeSeries, Sequence[TimeSeries]],
        validation_series: Union[TimeSeries, Sequence[TimeSeries]],
        validation_covariates: Union[TimeSeries, Sequence[TimeSeries]],
    ):
        predict_covariates = [
            past_covariate.append(validation_past_covariate)
            for past_covariate, validation_past_covariate in zip(
                covariates, validation_covariates
            )
        ]

        # y_train = pd.concat([s.pd_dataframe() for s in series], )
        # y_test = pd.concat([s.pd_dataframe() for s in validation_series], )
        # x_train = pd.concat([s.pd_dataframe() for s in covariates], )
        # x_test = pd.concat([s.pd_dataframe() for s in validation_covariates], )
        #
        # # We then fit the StandardScaler on the whole training set
        # sc = StandardScaler()
        # sc.fit(x_train)
        # X_train_scaled = sc.transform(
        #     x_train
        # )
        #
        # X_test_scaled = sc.transform(
        #     x_test
        # )

        def objective(trial: Trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
            }
            lags = trial.suggest_int("lags", 1, 30)
            model = RegressionModel(
                lags_future_covariates=(lags, 1),
                model=TransformedTargetRegressor(
                    regressor=ExtraTreesRegressor(random_state=42, **params),
                    func=log10p,
                    inverse_func=exp10p,
                ),
            )
            model.fit(
                series=series,
                future_covariates=covariates,
            )
            preds = model.predict(
                n=7,
                series=series,
                future_covariates=predict_covariates,
            )

            scores = mae(validation_series, preds, n_jobs=-1, verbose=True)
            score_val = np.mean(scores)

            return score_val if score_val != np.nan else float("inf")

        return objective

    def build_model(
        self, params: dict[str, Any], **kwargs
    ) -> GlobalForecastingModel:
        lags = params.pop("lags")
        return RegressionModel(
            lags_future_covariates=(lags, 1),
            model=TransformedTargetRegressor(
                regressor=ExtraTreesRegressor(**params),
                func=log10p,
                inverse_func=exp10p,
            ),
        )

    def name(self) -> str:
        return "ExtraTreesRegressor"


class GlobalRetrainingClusterModel(GlobalForecastingModel):
    def __init__(self, model: Pipeline):
        super().__init__()
        self.model = model

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
    ) -> "GlobalForecastingModel":
        # Check if the series is a sequence of TimeSeries
        if isinstance(series, Sequence):
            y_train = pd.concat([s.pd_dataframe() for s in series])
        else:
            y_train = series.pd_dataframe()

        if past_covariates is not None:
            raise NotImplementedError(
                "Past covariates are not supported for this model"
            )

        # Check if the covariates is a sequence of TimeSeries
        if isinstance(future_covariates, Sequence):
            x_train = pd.concat([s.pd_dataframe() for s in future_covariates])
        else:
            x_train = future_covariates.pd_dataframe()

        # We then fit the StandardScaler on the whole training set
        sc = StandardScaler()
        sc.fit(x_train)
        X_train_scaled = sc.transform(x_train)
        self.model.fit(X_train_scaled, y_train)

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
        # Check if the covariates is a sequence of TimeSeries
        if isinstance(future_covariates, Sequence) and isinstance(
            series, Sequence
        ):
            return [
                self._predict_single(n, s, None, c)
                for s, c in zip(series, future_covariates)
            ]
        return self._predict_single(
            n, series, past_covariates, future_covariates
        )

    def _predict_single(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        if past_covariates is not None:
            raise NotImplementedError(
                "Past covariates are not supported for this model"
            )

        # Check if the series is a sequence of TimeSeries
        if isinstance(series, TimeSeries):
            y_pred = self.model.predict(future_covariates.pd_dataframe())
            return TimeSeries.from_dataframe(
                pd.DataFrame(
                    y_pred,
                    index=[
                        series.end_time() + i * series.freq() for i in range(n)
                    ],
                    columns=[series.components[0]],
                )
            )
        else:
            raise NotImplementedError(
                "This model does not support multiple time series"
            )

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
        return None, None, None, None, None, None

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
        return None, None, False, False, None, None
