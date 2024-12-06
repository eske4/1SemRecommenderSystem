import numpy as np


class UserProfileBuilder:
    """
    A class to create user profiles from a ratings matrix and track features.

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
        Retrieves the list of track IDs rated by a specific user.

        Parameters:
            user_id (int): The user ID.
            ratings (pd.DataFrame): The ratings dataset.

        Returns:
            np.ndarray: An array of track IDs rated by the user.
        """
        user_ratings = ratings[ratings["user_id"] == user_id]
        return user_ratings["track_id"].values
