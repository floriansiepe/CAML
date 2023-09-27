import logging
from typing import List

import pandas as pd
from darts import TimeSeries
from darts.datasets import ElectricityDataset
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from src.aggregation.fitted_global_forecasting_model import FittedGlobalForecastingModel
from src.aggregation.model_aggregation import ModelAggregation
from src.aggregation.objective_factories.xgboost_objective_factory import (
    XGBoostObjectiveFactory,
)
from src.clustering.model_clustering import ModelClustering
from src.evaluation.timeseries_clustering_evaluator import TimeseriesClusteringEvaluator
from src.loader.data_loader import DataLoader
from src.loader.model_data import ModelData

# Set logging level to INFO to see some output during the clustering process
logging.basicConfig(level=logging.INFO)


class DummyDataLoader(DataLoader):
    def load(self) -> List[ModelData]:
        model_data_set = []
        # Load the univariate time series dataset
        multi_serie_elec = ElectricityDataset().load()
        # Drop some data to speed up the demo
        multi_serie_elec = multi_serie_elec.drop_before(pd.Timestamp("2012-01-01"))

        for i in range(0, 3):
            timeseries = multi_serie_elec.univariate_component(i + 1)

            # Split into test, validation, and training sets:
            train, val = timeseries.split_after(pd.Timestamp("2014-08-01"))
            val, test = val.split_after(pd.Timestamp("2014-12-01"))
            test = test[: len(val)]

            # Train a fast model
            model = XGBoostObjectiveFactory().build_model(params={})
            forecasting_model = FittedGlobalForecastingModel(model)
            forecasting_model.clever_fit(train, covariates=train)

            # Because the toy data set is univariate, features (x) and labels (y) are the same
            model_data = ModelData(
                id=i,
                model=forecasting_model,
                train_x=train,
                train_y=train,
                validation_x=val,
                validation_y=val,
                test_x=test,
                test_y=test,
            )
            model_data_set.append(model_data)

        return model_data_set


def predictor(
    model: GlobalForecastingModel, n: int, series: TimeSeries, covariates: TimeSeries
) -> TimeSeries:
    if type(model) == FittedGlobalForecastingModel:
        return model.clever_predict(n, series, covariates)
    return model.predict(n, series, future_covariates=covariates)


def create_cluster_models():
    logging.info("Loading the clustering model")
    data_loader = DummyDataLoader()
    data = data_loader.load()
    logging.info("Clustering the models")
    clustering = ModelClustering(data=data, predictor=predictor, verbose=True)
    clustering.fit()
    logging.info("Extracting cluster labels")
    labels = clustering.transform(k=2)

    logging.info("Aggregate models")
    model_aggregation = ModelAggregation(
        objective_factories=[XGBoostObjectiveFactory()]
    )
    model_aggregation.fit(model_data=data, cluster=labels)
    cluster_models_df = model_aggregation.transform()
    logging.info("Done")

    # Pickle the cluster models
    cluster_models_df.to_pickle("cluster_models.pkl")


def load_cluster_models():
    # Load the cluster models
    return pd.read_pickle("cluster_models.pkl")


def evaluate_cluster_models(cluster_models_df):
    logging.info("Evaluate the cluster models")
    clustering_evaluator = TimeseriesClusteringEvaluator(
        cluster_models_df, predictor=predictor
    )
    clustering_evaluator.evaluate()

    clustering_evaluator.get_scores().to_csv("cluster_models_scores.csv")
    clustering_evaluator.get_raw_scores().to_csv("cluster_models_raw_scores.csv")


create_cluster_models()
cluster_models = load_cluster_models()
evaluate_cluster_models(cluster_models)
