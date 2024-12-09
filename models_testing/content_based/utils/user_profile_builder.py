import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class UserProfileBuilder:

    def __init__(self):
        """Initialize the UserProfileBuilder class."""
        pass

    """
    ass to create user profiles from a ratings matrix and track features.

    This class aggregates user preferences, retrieves user interactions,
    and processes data to create profiles suitable for recommendation systems.
    """

    @staticmethod
    def aggregate_user_preference(user_id, ratings, tracks):
        """
        Aggregates a user's preferences based on their ratings and the associated track features.

        Parameters:
            user_id (int): The user ID to filter the tracks.
            ratings (pd.DataFrame): The ratings dataset containing user-item interactions.
            tracks (pd.DataFrame): The tracks dataset containing track features.

        Returns:
            np.ndarray: A numerical vector representing the aggregated user profile.
        """
        # Step 1: Get the track IDs that the user has rated
        user_ratings = ratings[ratings["user_id"] == user_id]
        user_track_ids = user_ratings["track_id"].values

        # Step 2: Select the track features corresponding to the rated tracks
        user_tracks = tracks.iloc[user_track_ids]

        # Step 3: Aggregate features (e.g., by averaging)
        return np.mean(user_tracks, axis=0)

    @staticmethod
    def aggregate_user_preference_with_weight(user_id, ratings, tracks):
        """
        Aggregates a user's preferences based on their ratings and the associated track features.

        Parameters:
            user_id (int): The user ID to filter the tracks.
            ratings (pd.DataFrame): The ratings dataset containing user-item interactions.
            tracks (pd.DataFrame): The tracks dataset containing track features.

        Returns:
            np.ndarray: A numerical vector representing the aggregated user profile.
        """
        # Filter user's ratings and corresponding tracks
        user_ratings = ratings[ratings["user_id"] == user_id]
        user_track_ids = user_ratings["track_id"].values

        # Ensure we match indices correctly to avoid errors
        user_tracks = tracks.loc[user_track_ids]

        # Scale and adjust playcount weights
        weights = MinMaxScaler().fit_transform(user_ratings[["playcount"]]) + 1

        # Apply weights to track features
        user_tracks_weighted = user_tracks.multiply(weights.flatten(), axis=0)

        # Scale the track features, preserving headers and indices
        scaler = MinMaxScaler()
        user_tracks_scaled = pd.DataFrame(
            scaler.fit_transform(user_tracks_weighted),
            columns=user_tracks.columns,
            index=user_tracks.index,
        )

        # Aggregate features by averaging across columns
        return user_tracks_scaled.mean(axis=0)

    @staticmethod
    def get_all_users(ratings):
        """
        Retrieves all unique user IDs from the ratings dataset.

        Parameters:
            ratings (pd.DataFrame): The ratings dataset.

        Returns:
            np.ndarray: An array of unique user IDs.
        """
        return ratings["user_id"].unique()

    @staticmethod
    def get_rated_list(user_id, ratings):
        """
        Retrieves the list of track IDs rated by a specific user and scales the playcount to relevancy.

        Parameters:
            user_id (int): The user ID.
            ratings (pd.DataFrame): The ratings dataset.

        Returns:
            pd.DataFrame: A DataFrame with the rated track IDs and their scaled relevancy scores.
        """
        # Retrieve ratings for the specific user and create a copy to avoid setting on a slice
        user_ratings = ratings[ratings["user_id"] == user_id].copy()

        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Scale the playcount (relevancy) between 0 and 1
        user_ratings["score"] = scaler.fit_transform(user_ratings[["playcount"]])

        # Reset the index to avoid the old index being included as a column
        user_ratings = user_ratings.reset_index(drop=True)

        # Return the rated tracks along with their relevancy
        return user_ratings[["track_id", "score"]]
