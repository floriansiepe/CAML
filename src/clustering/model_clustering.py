from typing import Callable, List

import fastcluster
import numpy as np
from darts import TimeSeries
from darts.metrics import mae
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from scipy.cluster.hierarchy import fcluster

from src.distance.distance import pdist_wrapper
from src.distance.pairwise_loss_distance import PairwiseLossDistance
from src.loader.model_data import ModelData


class ModelClustering:
    def __init__(
        self,
        data: List[ModelData],
        predictor: Callable[
            [GlobalForecastingModel, int, TimeSeries, TimeSeries], TimeSeries
        ],
        loss_function=mae,
        linkage_method="ward",
        verbose=False,
    ):
        self.data = data
        self.predictor = predictor
        self.loss_function = loss_function
        self.linkage_method = linkage_method
        self.linkage = None
        self.fitted = False
        self.verbose = verbose

    def fit(self):
        distance_function = PairwiseLossDistance(self.loss_function, self.predictor)
        distance_matrix = pdist_wrapper(
            self.data, distance_function.distance, verbose=self.verbose
        )
        self.linkage = fastcluster.linkage(
            distance_matrix,
            method=self.linkage_method,
        )
        self.fitted = True

    def fit_transform(self, k: int) -> np.ndarray:
        self.fit()
        return self.transform(k)

    def transform(self, k: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Must call fit or fit_transform before calling transform")
        return self._flat_clusters(k)

    def _flat_clusters(self, k: int) -> np.ndarray:
        return fcluster(self.linkage, t=k, criterion="maxclust")
