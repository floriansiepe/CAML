from typing import List

from src.aggregation.model_aggregation import ModelAggregation
from src.aggregation.objective_factories.xgboost_objective_factory import XGBoostObjectiveFactory
from src.clustering.model_clustering import ModelClustering
from src.loader.data_loader import DataLoader
from src.loader.model_data import ModelData


class DummyDataLoader(DataLoader):
    def load(self) -> List[ModelData]:
        return []


data_loader = DummyDataLoader()
clustering = ModelClustering(data_loader=data_loader)
clustering.fit()
labels = clustering.transform(k=42)

objective_factory = XGBoostObjectiveFactory()
model_aggregation = ModelAggregation(objective_factories=[objective_factory])
model_aggregation.fit(model_data=data_loader.load(), cluster=labels)
cluster_models = model_aggregation.transform()
