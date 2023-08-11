from typing import List, Union

import numpy as np
import pandas as pd
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from src.aggregation.cluster_model_trainer import ClusterModelTrainer
from src.aggregation.objective_factories.objective_factory import ObjectiveFactory
from src.loader.model_data import ModelData


class ModelAggregation:
    def __init__(self, objective_factories: List[ObjectiveFactory]):
        self.df = None
        self.objective_factories = objective_factories
        self.fitted = False

    def fit(self, model_data: List[ModelData], cluster: Union[np.ndarray, List[int]]):
        if len(cluster) == 0:
            raise ValueError("Cluster must not be empty")
        if len(model_data) != len(cluster):
            raise ValueError("Cluster must have the same length as model_data")

        # Create a dataframe with model_data and cluster as columns
        self.df = pd.DataFrame({
            "model_data": model_data,
            "cluster": cluster
        })

        cluster_model_data = []

        for cluster_id in self.df["cluster"].unique():
            cluster_dataset = self.df[self.df["cluster"] == cluster_id]["model_data"].tolist()
            cluster_model = self._aggregate(cluster_dataset, cluster_id)
            cluster_model_data.append(cluster_model)

        self.df["cluster_model"] = self.df["cluster"].apply(lambda x: cluster_model_data[x])
        self.fitted = True

    def transform(self) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("ModelAggregation must be fitted before transform")

        return self.df

    def fit_transform(self, model_data: List[ModelData], cluster: List[int]) -> pd.DataFrame:
        self.fit(model_data, cluster)
        return self.transform()

    def _aggregate(self, cluster_dataset: List[ModelData], cluster_id: int) -> GlobalForecastingModel:
        cluster_model_trainer = ClusterModelTrainer(self.objective_factories)
        return cluster_model_trainer.train(
            cluster_dataset, cluster_id=cluster_id
        )
