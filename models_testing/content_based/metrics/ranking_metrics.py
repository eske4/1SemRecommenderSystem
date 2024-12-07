import numpy as np
import pandas as pd


class RankingMetrics:
    """
    A class to calculate ranking metrics for a single user.
    """

    def __init__(self, recommended=None, relevant=None):
        """
        Initialize the metrics calculator and compute all metrics.

        Parameters:
            recommended (list): List of recommended item IDs (ordered by relevance).
            relevant (pd.DataFrame): DataFrame with columns 'track_id' (item IDs) and 'score' (relevance scores).
        """
        self.recommended = recommended if recommended is not None else []
        self.relevant = (
            relevant
            if relevant is not None
            else pd.DataFrame(columns=["track_id", "score"])
        )
        self.k = len(self.recommended)
        self.metrics_count = 0

        if self.k > 0 and not self.relevant.empty:
            # Precompute all metrics
            self.precision_k = self.precision_at_k(self.k)
            self.recall_k = self.recall_at_k(self.k)
            self.ndcg_k = self.ndcg_at_k(self.k)
            self.hit_k = self.hit_at_k(self.k)
            self.map_k = self.mean_average_precision()
            self.metrics_count += 1
        else:
            # Set all metrics to 0
            self.precision_k = 0
            self.recall_k = 0
            self.ndcg_k = 0
            self.hit_k = 0
            self.map_k = 0

    def __add__(self, other):
        """
        Add two RankingMetrics instances by aggregating their metrics.

        Parameters:
            other (RankingMetrics): Another instance of RankingMetrics.

        Returns:
            RankingMetrics: A new instance with aggregated metrics.
        """
        if not isinstance(other, RankingMetrics):
            raise ValueError("Can only add instances of RankingMetrics.")

        # Create a new aggregated instance with empty recommended and relevant
        new_instance = RankingMetrics(
            recommended=[], relevant=pd.DataFrame(columns=["track_id", "score"])
        )

        # Aggregate metrics
        new_instance._set_aggregated_metrics(
            precision_k=self.precision_k + other.precision_k,
            recall_k=self.recall_k + other.recall_k,
            ndcg_k=self.ndcg_k + other.ndcg_k,
            hit_k=self.hit_k + other.hit_k,
            map_k=self.map_k + other.map_k,
        )

        # Combine metrics_count
        new_instance.metrics_count = self.metrics_count + other.metrics_count

        return new_instance

    def _set_aggregated_metrics(self, precision_k, recall_k, ndcg_k, hit_k, map_k):
        """
        Set precomputed metrics for an aggregated instance.
        """
        self.precision_k = precision_k
        self.recall_k = recall_k
        self.ndcg_k = ndcg_k
        self.hit_k = hit_k
        self.map_k = map_k
        return self

    def precision_at_k(self, k):
        if k == 0:
            return 0.0
        top_k = self.recommended[:k]
        relevant_items = set(self.relevant["track_id"].values)
        relevant_count = sum(1 for item in top_k if item in relevant_items)
        return relevant_count / k

    def recall_at_k(self, k):
        if self.relevant.empty:
            return 0.0
        top_k = self.recommended[:k]
        relevant_items = set(self.relevant["track_id"].values)
        relevant_count = sum(1 for item in top_k if item in relevant_items)
        return relevant_count / len(relevant_items)

    def ndcg_at_k(self, k):
        if self.relevant.empty:
            return 0.0
        top_k = self.recommended[:k]
        relevant_dict = dict(zip(self.relevant["track_id"], self.relevant["score"]))
        dcg = sum(
            relevant_dict.get(item, 0) / np.log2(idx + 2)
            for idx, item in enumerate(top_k)
        )
        ideal_dcg = sum(
            score / np.log2(idx + 2)
            for idx, score in enumerate(
                sorted(self.relevant["score"], reverse=True)[:k]
            )
        )
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def hit_at_k(self, k):
        top_k = self.recommended[:k]
        relevant_items = set(self.relevant["track_id"].values)
        return 1.0 if any(item in relevant_items for item in top_k) else 0.0

    def average_precision(self):
        if self.relevant.empty:
            return 0.0

        precision_sum = 0.0
        relevant_count = 0
        relevant_items = set(self.relevant["track_id"].values)

        for i, item in enumerate(self.recommended, start=1):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / i

        return precision_sum / len(relevant_items)

    def mean_average_precision(self):
        return self.average_precision()

    def metrics_summary(self):
        """
        Get a summary of all precomputed metrics.

        Returns:
            dict: Dictionary containing Precision@k, Recall@k, NDCG@k, Hit@k, and MAP.
        """
        if self.metrics_count == 0:
            return {
                "Precision@k": 0.0,
                "Recall@k": 0.0,
                "NDCG@k": 0.0,
                "Hit@k": 0.0,
                "MAP": 0.0,
            }

        return {
            "Precision@k": self.precision_k / self.metrics_count,
            "Recall@k": self.recall_k / self.metrics_count,
            "NDCG@k": self.ndcg_k / self.metrics_count,
            "Hit@k": self.hit_k / self.metrics_count,
            "MAP": self.map_k / self.metrics_count,
        }
