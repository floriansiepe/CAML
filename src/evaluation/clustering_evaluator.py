from abc import ABC, abstractmethod


class ClusteringEvaluator(ABC):
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError(
            "ClusteringEvaluator is an abstract class and cannot be instantiated."
        )

    @abstractmethod
    def get_clusters(self):
        raise NotImplementedError(
            "ClusteringEvaluator is an abstract class and cannot be instantiated."
        )

    @abstractmethod
    def get_scores(self):
        raise NotImplementedError(
            "ClusteringEvaluator is an abstract class and cannot be instantiated."
        )

    @abstractmethod
    def get_raw_scores(self):
        raise NotImplementedError(
            "ClusteringEvaluator is an abstract class and cannot be instantiated."
        )