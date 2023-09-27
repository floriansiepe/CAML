import logging
import warnings
from typing import List

import optuna
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from src.aggregation.fitted_global_forecasting_model import FittedGlobalForecastingModel
from src.aggregation.objective_factories.objective_factory import ObjectiveFactory
from src.loader.model_data import ModelData

warnings.simplefilter("ignore", UserWarning)


# for convenience, print some optimization trials information
def print_callback(study, trial):
    logging.info(f"Current value: {trial.value}, Current params: {trial.params}")
    logging.info(
        f"Best value: {study.best_value}, Best params: {study.best_trial.params}"
    )


class ClusterModelTrainer:
    def __init__(self, objective_factories: List[ObjectiveFactory]):
        self.objective_factory = objective_factories

    def train(
        self,
        timeseries_dataset: List[ModelData],
        verbose: bool = True,
        global_fit: bool = True,
        **kwargs,
    ) -> GlobalForecastingModel:
        """Train a cluster model"""
        logging.info("Training cluster model. size: %s", len(timeseries_dataset))
        pipeline_series = Pipeline(
            [
                Scaler(n_jobs=-1, verbose=verbose, global_fit=global_fit),
            ]
        )
        pipeline_covariates = Pipeline(
            [
                Scaler(n_jobs=-1, verbose=verbose, global_fit=global_fit),
            ]
        )
        series = []
        covariates_retrain = []
        validation_series = []
        validation_covariates = []
        test_series = []
        test_covariates = []

        for split_timeseries in timeseries_dataset:
            series.append(split_timeseries.train_y)
            covariates_retrain.append(split_timeseries.train_x)
            validation_series.append(split_timeseries.validation_y)
            validation_covariates.append(split_timeseries.validation_x)
            test_series.append(split_timeseries.test_y)
            test_covariates.append(split_timeseries.test_x)

        # Fit the pipeline on the training data
        series_scaled = pipeline_series.fit_transform(series)
        covariates_scaled = pipeline_covariates.fit_transform(covariates_retrain)
        validation_series_scaled = pipeline_series.transform(validation_series)
        validation_covariates_scaled = pipeline_covariates.transform(
            validation_covariates
        )

        scores = []
        params = []

        # Iterate over the objective factories and find the best hyperparameters
        for i, objective_factory in enumerate(self.objective_factory):
            logging.info("Training cluster model. objective_factory: %s", i)
            # Train the model on the training data
            objective = objective_factory.create(
                series_scaled,
                covariates_scaled,
                validation_series_scaled,
                validation_covariates_scaled,
            )

            # optimize hyperparameters by minimizing on the validation set
            study = optuna.create_study(direction="minimize")
            study.optimize(
                objective,
                n_trials=1,
                callbacks=[print_callback] if verbose else [],
                timeout=30,
                n_jobs=-1,
            )
            best_params = study.best_params
            best_value = study.best_value
            scores.append(best_value)
            params.append(best_params)

        # Select the best objective factory
        best_objective_factory = self.objective_factory[scores.index(min(scores))]

        # Select the best hyperparameters
        best_params = params[scores.index(min(scores))]

        # Build the model with the best hyperparameters
        # Re-fit the pipeline on the whole dataset
        series_retrain = []
        covariates_retrain = []
        val_series_retrain = []
        val_covariates_retrain = []
        for split_timeseries in timeseries_dataset:
            series = split_timeseries.train_y.append(split_timeseries.validation_y)
            covariate = split_timeseries.train_x.append(split_timeseries.validation_x)
            series_retrain.append(series)
            covariates_retrain.append(covariate)
            val_series_retrain.append(series.append(split_timeseries.test_y))
            val_covariates_retrain.append(covariate.append(split_timeseries.test_x))

        best_model = best_objective_factory.build_model(best_params, **kwargs)

        cluster_model = FittedGlobalForecastingModel(best_model)

        # Train the model on the whole dataset
        cluster_model.clever_fit(
            series=series_retrain,
            covariates=covariates_retrain,
            val_series=val_series_retrain,
            val_covariates=val_covariates_retrain,
            epochs=1,
        )

        return cluster_model
