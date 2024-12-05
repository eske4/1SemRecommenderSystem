import numpy as np


class RankingMetrics:
    """
    A class to calculate ranking metrics for a single user.
    """

    def __init__(self, recommended, relevant):
        """
        Initialize the metrics calculator.

        Parameters:
            recommended (list): List of recommended item IDs (ordered by relevance).
            relevant (set): Set of relevant item IDs.
        """
        self.recommended = recommended
        self.relevant = set(relevant)

    def precision_at_k(self, k):
        """
        Calculate Precision@k.

        Parameters:
            k (int): Number of top items to consider.

        Returns:
            float: Precision@k score.
        """
        if k == 0:
            return 0.0
        top_k = self.recommended[:k]
        relevant_count = sum(1 for item in top_k if item in self.relevant)
        return relevant_count / k

    def recall_at_k(self, k):
        """
        Calculate Recall@k.

        Parameters:
            k (int): Number of top items to consider.

        Returns:
            float: Recall@k score.
        """
        if not self.relevant:
            return 0.0
        top_k = self.recommended[:k]
        relevant_count = sum(1 for item in top_k if item in self.relevant)
        return relevant_count / len(self.relevant)

    def ndcg_at_k(self, k):
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)@k.

        Parameters:
            k (int): Number of top items to consider.

        Returns:
            float: NDCG@k score.
        """
        top_k = self.recommended[:k]
        dcg = sum(
            int(item in self.relevant)
            / np.log2(idx + 2)  # idx + 2 because log2(1) is undefined
            for idx, item in enumerate(top_k)
        )
        ideal_dcg = sum(
            1 / np.log2(idx + 2) for idx in range(min(len(self.relevant), k))
        )
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def hit_at_k(self, k):
        """
        Calculate Hit@k.

        Parameters:
            k (int): Number of top items to consider.

        Returns:
            float: Hit@k score (1.0 if any relevant item is in the top-k, otherwise 0.0).
        """
        top_k = self.recommended[:k]
        return 1.0 if any(item in self.relevant for item in top_k) else 0.0

    def average_precision(self):
        """
        Calculate Average Precision (AP).

        Returns:
            float: Average Precision score.
        """
        if not self.relevant:
            return 0.0

        precision_sum = 0.0
        relevant_count = 0

        for i, item in enumerate(self.recommended, start=1):
            if item in self.relevant:
                relevant_count += 1
                precision_sum += relevant_count / i

        return precision_sum / len(self.relevant)

    def mean_average_precision(self):
        """
        Calculate Mean Average Precision (MAP).

        Returns:
            float: Mean Average Precision score.
        """
        return self.average_precision()

    def metrics_summary(self, k):
        """
        Get a summary of all metrics at a specific k.

        Parameters:
            k (int): Number of top items to consider.

        Returns:
            dict: Dictionary containing Precision@k, Recall@k, NDCG@k, Hit@k, and MAP.
        """
        return {
            "Precision@k": self.precision_at_k(k),
            "Recall@k": self.recall_at_k(k),
            "NDCG@k": self.ndcg_at_k(k),
            "Hit@k": self.hit_at_k(k),
            "MAP": self.mean_average_precision(),
        }
