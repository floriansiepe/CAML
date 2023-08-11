from typing import Tuple

import pandas as pd

from src.aggregation.fitted_global_forecasting_model import FittedGlobalForecastingModel
from src.evaluation.clustering_evaluator import ClusteringEvaluator
from src.evaluation.timeseries_model_evaluator import TimeseriesEvaluator


class TimeseriesClusteringEvaluator(ClusteringEvaluator):
    def __init__(
            self,
            cluster_model_df: pd.DataFrame,
    ):
        if cluster_model_df is None:
            raise ValueError("model_df must be provided")
        self.model_df = cluster_model_df
        # Prepare the scores dataframe
        self.scores = pd.DataFrame(
            columns=[
                "cluster",
                "cluster_size",
                "model_mae",
                "model_mse",
                "model_rmse",
                "model_uncertainty",
                "model_smape",
                "cluster_mae",
                "cluster_mse",
                "cluster_rmse",
                "cluster_uncertainty",
                "cluster_smape",
            ]
        )

        self.raw_scores = []
        self.timeseries = {}
        self.cluster_models = {}
        for _, row in cluster_model_df.iterrows():
            self.timeseries[row["model_data"].id] = row["model_data"]
        for cluster in cluster_model_df["cluster"].unique():
            self.cluster_models[cluster] = cluster_model_df[cluster_model_df["cluster"] == cluster]["model"].iloc[0]

    def evaluate(self):
        self._evaluate_scores()

    def _evaluate_scores(self):
        # Evaluate the cluster and baseline scores for each cluster
        series = self.model_df.groupby("cluster").apply(
            self._evaluate_cluster_scores
        )
        columns = [
            "cluster",
            "cluster_size",
            "model_mae",
            "model_mse",
            "model_rmse",
            "model_uncertainty",
            "model_smape",
            "cluster_mae",
            "cluster_mse",
            "cluster_rmse",
            "cluster_uncertainty",
            "cluster_smape",
        ]
        self.scores = pd.DataFrame(
            [
                [a, b, c, d, e, f, g, h, i, j, k, l]
                for a, b, c, d, e, f, g, h, i, j, k, l in series.values
            ],
            columns=columns,
        )

        for score in ["mae", "mse", "rmse", "uncertainty", "smape"]:
            self.scores[f"normalized_model_{score}"] = 0
            self.scores[f"normalized_cluster_{score}"] = (
                    self.scores[f"cluster_{score}"] - self.scores[f"model_{score}"]
            )

    def _evaluate_cluster_scores(self, cluster_df: pd.DataFrame):
        # Build a cluster model
        cluster = cluster_df["cluster"].iloc[0]
        cluster_model = self.cluster_models[cluster]

        # Evaluate the cluster and baseline model on the baseline model's test set
        cluster_df[
            [
                "model_mae",
                "model_mse",
                "model_rmse",
                "model_uncertainty",
                "model_smape",
                "cluster_mae",
                "cluster_mse",
                "cluster_rmse",
                "cluster_uncertainty",
                "cluster_smape",
            ]
        ] = cluster_df.apply(
            lambda row: self._evaluate_prediction(row, cluster_model),
            axis=1,
            result_type="expand",
        )

        raw_scores = cluster_df[
            [
                "model_mae",
                "model_mse",
                "model_rmse",
                "model_uncertainty",
                "model_smape",
                "cluster_mae",
                "cluster_mse",
                "cluster_rmse",
                "cluster_uncertainty",
                "cluster_smape",
            ]
        ].copy()
        raw_scores["cluster"] = cluster
        raw_scores["cluster_size"] = len(cluster_df)

        self.raw_scores.append(raw_scores)

        # Aggregate the scores by taking the mean
        return (
            cluster,
            len(cluster_df),
            cluster_df["model_mae"].mean(),
            cluster_df["model_mse"].mean(),
            cluster_df["model_rmse"].mean(),
            cluster_df["model_uncertainty"].mean(),
            cluster_df["model_smape"].mean(),
            cluster_df["cluster_mae"].mean(),
            cluster_df["cluster_mse"].mean(),
            cluster_df["cluster_rmse"].mean(),
            cluster_df["cluster_uncertainty"].mean(),
            cluster_df["cluster_smape"].mean(),
        )

    def _evaluate_prediction(
            self, row, cluster_model: FittedGlobalForecastingModel
    ) -> Tuple[
        float, float, float, float, float, float, float, float, float, float
    ]:
        model = row["model_data"].model
        return self._evaluate_model(row, model, cluster_model)

    def _evaluate_model(self, row, baseline, cluster_model: FittedGlobalForecastingModel):
        (
            model_mae,
            model_mse,
            model_rmse,
            model_uncertainty,
            model_smape,
        ) = TimeseriesEvaluator(row, baseline, self.timeseries).evaluate()
        (
            cluster_model_mae,
            cluster_model_mse,
            cluster_model_rmse,
            cluster_model_uncertainty,
            cluster_model_smape,
        ) = TimeseriesEvaluator(row, cluster_model, self.timeseries).evaluate()
        return (
            model_mae,
            model_mse,
            model_rmse,
            model_uncertainty,
            model_smape,
            cluster_model_mae,
            cluster_model_mse,
            cluster_model_rmse,
            cluster_model_uncertainty,
            cluster_model_smape,
        )

    def get_clusters(self):
        return self.model_df["cluster"]

    def get_scores(self):
        return self.scores

    def get_raw_scores(self):
        return pd.concat(self.raw_scores)

    def get_mean_cluster_mae(self):
        return self.scores["cluster_mae"].mean()

    def get_mean_model_mae(self):
        return self.scores["model_mae"].mean()

    def get_normalized_mae(self):
        return self.get_mean_cluster_mae() - self.get_mean_model_mae()

    def _n_clusters_valid(self):
        n_clusters = len(self.model_df["cluster"].unique())
        n_samples = len(self.model_df)
        return 1 < n_clusters < n_samples
