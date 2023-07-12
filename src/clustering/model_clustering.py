import fastcluster
import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import mean_absolute_error

from src.distance.distance import pdist_wrapper
from src.distance.pairwise_loss_distance import PairwiseLossDistance
from src.loader.data_loader import DataLoader


class ModelClustering:
    def __init__(self, data_loader: DataLoader, loss_function=mean_absolute_error, linkage_method="ward"):
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.linkage_method = linkage_method
        self.linkage = None
        self.fitted = False

    def fit(self):
        distance_function = PairwiseLossDistance(self.loss_function)
        data = self.data_loader.load()
        distance_matrix = pdist_wrapper(data, distance_function.distance)
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
        return fcluster(
            self.linkage,
            t=k,
            criterion='maxclust'
        )
