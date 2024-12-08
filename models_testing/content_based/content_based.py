import os

import numpy as np
import pandas as pd
from custom_recommenders.autoencode_recommender import AutoencodeRecommender
from metrics.diversity_metrics import DiversityMetrics
from metrics.ranking_metrics import RankingMetrics
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from utils.user_profile_builder import UserProfileBuilder

# Change working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    return pd.read_csv(
        "../../remappings/data/Modified_Music_info.txt", delimiter="\t"
    ), pd.read_csv(
        "../../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt",
        delimiter="\t",
    )


def preprocess_data(data, scaler=MinMaxScaler()):
    meta = data[["name", "artist", "track_id"]]
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

    return processed_data, meta


def main():

    # Load and process the data
    tracks, ratings = load_data()
    tracks, meta = preprocess_data(tracks)

    # Initialize autoencoder recommender
    autoencode_recommender = AutoencodeRecommender(
        data=tracks, latent_dim=90, meta_data=meta
    )

    # Get 100 users for testings
    user_ids = UserProfileBuilder.get_all_users(ratings)[:100]

    # prepare for ranking
    mean_ranking = RankingMetrics()
    diversity_count = 0
    mean_diversity = 0

    for user in user_ids:

        top_n = 10
        # Get the average profile of the user tracks
        input_feature = UserProfileBuilder.aggregate_user_preference(
            user, ratings, autoencode_recommender.encoded_data
        )

        # Get the list of the users listened tracks
        user_ratings = UserProfileBuilder.get_rated_list(user, ratings)

        # Recommend with autoencoder
        indices, scores, similar_items = autoencode_recommender.recommend_similar_items(
            input_feature, top_n=top_n
        )

        # Calculate diversity_score AILD
        diversity_score = DiversityMetrics.average_intra_list_distance(
            pd.DataFrame(similar_items).drop(columns=["name", "artist", "track_id"])
        )
        print(f"Diversity score: {diversity_score}")

        # Add user to the ranking metrics and display
        ranking = RankingMetrics(indices, user_ratings)
        mean_ranking += ranking
        diversity_count += 1
        mean_diversity += diversity_score
        print(f"user {user} Metrics Summary@10: {ranking.metrics_summary()}")

    # Print the final score
    print(f"mean rating Metrics Summary@10: {mean_ranking.metrics_summary()}")
    print(f"mean diversity score: {mean_diversity/diversity_count}")


if __name__ == "__main__":
    main()
