import os

import numpy as np
import pandas as pd
from custom_recommenders.autoencode_recommender import AutoencodeRecommender
from custom_recommenders.softmax_recommender import SoftmaxRecommender
from metrics.diversity_metrics import DiversityMetrics
from metrics.ranking_metrics import RankingMetrics
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, StandardScaler
from utils.user_profile_builder import UserProfileBuilder

# Change working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    return pd.read_csv(
        "../../remappings/data/Modified_Music_info.txt",
        delimiter="\t",
    ), pd.read_csv(
        "../../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt",
        delimiter="\t",
    )


def preprocess_data(data, scaler=MinMaxScaler()):
    meta = data[["track_id"]]
    features = data.drop(columns=["name", "artist", "track_id"], errors="ignore")
    features["genre"] = features["genre"].astype("category")

    # Handle tags, ensuring that we don't attempt to split non-string values
    features["tags"] = features["tags"].apply(
        lambda x: (
            eval(x)
            if isinstance(x, str) and x.startswith("[")
            else ([tag.strip() for tag in x.split(",")] if isinstance(x, str) else [])
        )
    )

    # Initialize MultiLabelBinarizer and fit it to the tags data
    mlb = MultiLabelBinarizer()
    tags_encoded = pd.DataFrame(
        mlb.fit_transform(features["tags"]), columns=mlb.classes_
    ).add_prefix("tag_")

    genre_encoded = pd.get_dummies(features["genre"], prefix="genre")

    # Combine the processed data
    processed_data = pd.concat([features, genre_encoded, tags_encoded], axis=1)
    processed_data = processed_data.drop(
        columns=["genre", "tags", "year"], errors="ignore"
    )

    # Scale the features
    processed_data = pd.DataFrame(
        scaler.fit_transform(processed_data), columns=processed_data.columns
    )
    merged_data = pd.merge(meta, processed_data, left_index=True, right_index=True)

    return processed_data, meta, merged_data


def main():
    # Load and process the data
    track, ratings = load_data()
    tracks, meta, tracks_with_id = preprocess_data(track, StandardScaler())
    # Initialize softmax recommender
    softmax_recommender = SoftmaxRecommender(tracks_with_id)

    # Get 100 users for testings
    user_ids = UserProfileBuilder.get_all_users(ratings)[:100]

    # prepare for ranking
    mean_ranking = RankingMetrics()

    for user in user_ids:

        top_n = 10
        # Get the average profile of the user tracks
        input_feature = UserProfileBuilder.aggregate_user_preference(
            user, ratings, tracks
        )

        # Get the list of the users listened tracks
        user_ratings = UserProfileBuilder.get_rated_list(user, ratings)

        # Recommend with autoencoder
        indices = softmax_recommender.recommend(input_feature, top_n)

        # Select rows where 'track_id' matches the ones in indices
        filtered_df = tracks_with_id[tracks_with_id["track_id"].isin(indices)]

        # Drop the 'track_id' column to get just the feature columns
        features_df = filtered_df.drop(columns=["track_id"])

        # Convert the features to a list of lists (each list contains the features for one track)
        indices_features = features_df.values.tolist()

        # Calculate diversity_score AILD
        diversity_score = DiversityMetrics.average_intra_list_distance(
            pd.DataFrame(indices_features)
        )
        print(f"Diversity score: {diversity_score}")

        # Add user to the ranking metrics and display
        ranking = RankingMetrics(indices, user_ratings)
        mean_ranking += ranking
        print(f"user {user} Metrics Summary@10: {ranking.metrics_summary()}")

    # Print the final score
    print(f"mean rating Metrics Summary@10: {mean_ranking.metrics_summary()}")


if __name__ == "__main__":
    main()
