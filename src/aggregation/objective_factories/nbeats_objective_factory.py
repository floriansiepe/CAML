from typing import Union, Sequence, Callable, Any

import numpy as np
from darts import TimeSeries
from darts.metrics import mae
from darts.models import NBEATSModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from optuna import Trial
from pytorch_lightning.callbacks import EarlyStopping

from src.aggregation.objective_factories.objective_factory import ObjectiveFactory
from src.aggregation.utils.pytorch_lightning_pruning_callback import PyTorchLightningPruningCallback


class NBeatsObjectiveFactory(ObjectiveFactory):
    def create(
            self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            covariates: Union[TimeSeries, Sequence[TimeSeries]],
            validation_series: Union[TimeSeries, Sequence[TimeSeries]],
            validation_covariates: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Callable[[Trial], float]:
        predict_covariates = [
            past_covariate.append(validation_past_covariate)
            for past_covariate, validation_past_covariate in zip(
                covariates, validation_covariates
            )
        ]

        val_series = [s.append(vs) for s, vs in zip(series, validation_series)]

        def objective(trial: Trial):
            # Suggest some hyperparameters for the XGBoost model
            params = {
                "input_chunk_length": trial.suggest_int("input_chunk_length", 30, 200),
                "num_stacks": trial.suggest_int("num_stacks", 2, 50),
                "num_blocks": trial.suggest_int("num_blocks", 1, 5),
                "num_layers": trial.suggest_int("num_layers", 2, 6),
                "dropout": trial.suggest_float("dropout", 0.0, 0.2),
                "n_epochs": trial.suggest_int("n_epochs", 1, 10),
            }

            # throughout training we'll monitor the validation loss for early stopping
            callbacks = [
                EarlyStopping(
                    "val_loss", min_delta=0.001, patience=3, verbose=True
                ),
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            ]

            pl_trainer_kwargs = {"callbacks": callbacks}

            # Generate a random id string
            model = NBEATSModel(
                output_chunk_length=1,
                random_state=42,
                pl_trainer_kwargs=pl_trainer_kwargs,
                n_epochs=params["n_epochs"],
                **params
            )
            model.fit(
                series=series,
                past_covariates=covariates,
                val_series=val_series,
                val_past_covariates=predict_covariates,
                epochs=params["n_epochs"],
                verbose=True,
            )

            preds = model.predict(
                n=7,
                series=series,
                past_covariates=predict_covariates,
                verbose=True,
            )

            scores = mae(validation_series, preds, n_jobs=-1, verbose=True)
            score_val = np.mean(scores)

            return score_val if score_val != np.nan else float("inf")

        return objective

    def build_model(
            self, params: dict[str, Any], **kwargs
    ) -> GlobalForecastingModel:
        n_epochs = kwargs.pop("n_epochs")

        model = NBEATSModel(
            output_chunk_length=1,
            random_state=42,
            n_epochs=n_epochs,
            save_checkpoints=True,
            force_reset=True,
            **params
        )
        return model

    def name(self) -> str:
        return "NBeats"
