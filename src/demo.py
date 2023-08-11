import logging
from typing import List

import pandas as pd
from darts import TimeSeries
from darts.datasets import ElectricityDataset
from darts.models import RegressionModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from src.aggregation.model_aggregation import ModelAggregation
from src.aggregation.objective_factories.lightgbm_objective_factory import LightGBMObjectiveFactory
from src.aggregation.objective_factories.xgboost_objective_factory import XGBoostObjectiveFactory
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
        multi_serie_elec = multi_serie_elec.drop_before(pd.Timestamp('2012-01-01'))

        for i in range(0, 10):
            timeseries = multi_serie_elec.univariate_component(i + 1)

            # Split into test, validation, and training sets:
            train, val = timeseries.split_after(pd.Timestamp('2014-08-01'))
            val, test = val.split_after(pd.Timestamp('2014-12-01'))
            test = test[:len(val)]

            # Train a fast model
            model = RegressionModel(lags=4)
            model.fit(train)

            # Because the toy data set is univariate, features (x) and labels (y) are the same
            model_data = ModelData(
                id=i,
                model=model,
                train_x=train,
                train_y=train,
                validation_x=val,
                validation_y=val,
                test_x=test,
                test_y=test,
            )
            model_data_set.append(model_data)

        return model_data_set


def predictor(model: GlobalForecastingModel, n: int, x: TimeSeries) -> TimeSeries:
    return model.predict(n, x)


logging.info("Loading the clustering model")
data_loader = DummyDataLoader()
data = data_loader.load()
logging.info("Clustering the models")
clustering = ModelClustering(data=data, predictor=predictor, verbose=True)
clustering.fit()
logging.info("Extracting cluster labels")
labels = clustering.transform(k=5)

logging.info("Aggregate models")
model_aggregation = ModelAggregation(objective_factories=[XGBoostObjectiveFactory(), LightGBMObjectiveFactory()])
model_aggregation.fit(model_data=data, cluster=labels)
cluster_models_df = model_aggregation.transform()
logging.info("Done")

cluster_models_df.to_parquet("cluster_models.parquet")

logging.info("Evaluate the cluster models")
clustering_evaluator = TimeseriesClusteringEvaluator(cluster_models_df)
