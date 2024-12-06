import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean


class DiversityMetrics:
    """
    A class to calculate diversity metrics for a list of items.

    Current metrics:
    - Average Intra-List Distance (AILD): Measures the diversity of a list
      based on the average pairwise distance between items.
    """

    def __init__(self):
        """Initialize the DiversityMetrics class."""
        pass

    @staticmethod
    def average_intra_list_distance(items, distance_metric="cosine"):
        """
        Calculate the Average Intra-List Distance (AILD).

        Parameters:
            items (pd.DataFrame or np.ndarray): A DataFrame or array of items (vectors).
            distance_metric (str): The distance metric to use ('euclidean' or 'cosine').

        Returns:
            float: The AILD score.
        """
        # Convert items to NumPy array if it's a DataFrame
        if isinstance(items, pd.DataFrame):
            items = items.to_numpy()

        n = len(items)
        if n < 2:
            return 0  # AILD is undefined for lists with fewer than 2 items

        # Select the appropriate distance function
        if distance_metric == "euclidean":
            dist_func = euclidean
        elif distance_metric == "cosine":
            dist_func = cosine
        else:
            raise ValueError(
                "Unsupported distance metric. Choose 'euclidean' or 'cosine'."
            )

        # Compute all pairwise distances
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = dist_func(items[i], items[j])  # Correctly access rows as vectors
                distances.append(dist)

        # Return the average distance as the overall diversity score
        return np.mean(distances)
