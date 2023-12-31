from typing import Sequence, Union, Any

import numpy as np
from darts import TimeSeries
from darts.metrics import mae
from darts.models import LightGBMModel
from optuna import Trial

from src.aggregation.objective_factories.objective_factory import ObjectiveFactory


class LightGBMObjectiveFactory(ObjectiveFactory):
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

        def objective(trial: Trial):
            # Suggest some hyperparameters for the LightGBM model
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0),
            }

            model = LightGBMModel(
                lags_future_covariates=(14, 1),
                output_chunk_length=1,
                random_state=42,
                **params
            )
            model.fit(
                series=series,
                future_covariates=covariates,
                verbose=True,
            )
            preds = model.predict(
                n=7,
                series=series,
                verbose=True,
                future_covariates=predict_covariates,
            )

            scores = mae(validation_series, preds, n_jobs=-1, verbose=True)
            score_val = np.mean(scores)

            return score_val if score_val != np.nan else float("inf")

        return objective

    def build_model(self, params: dict[str, Any], **kwargs) -> LightGBMModel:
        return LightGBMModel(
            lags_future_covariates=(14, 1),
            output_chunk_length=1,
            random_state=42,
            **params
        )

    def name(self) -> str:
        return "LightGBM"
