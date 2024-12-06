import numpy as np
import pandas as pd
from autoencode_recommender import AutoencodeRecommender
from diversity_metrics import DiversityMetrics
from ranking_metrics import RankingMetrics
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from user_profile_builder import UserProfileBuilder


def load_data():
    tracks_path = "../../remappings/data/Modified_Music_info.txt"
    ratings_path = "../../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt"
    tracks = pd.read_csv(tracks_path, delimiter="\t")
    ratings = pd.read_csv(ratings_path, delimiter="\t")
    return tracks, ratings


def preprocess_data(data, scaler=MinMaxScaler()):
    metadata = data[["name", "artist", "track_id"]]
    features = data.drop(columns=["name", "artist", "track_id"], errors="ignore")
    features["genre"] = features["genre"].astype("category")

    features["tags"] = features["tags"].apply(
        lambda x: (
            eval(x)
            if isinstance(x, str) and x.startswith("[") and x.endswith("]")
            else (
                [tag.strip() for tag in x.split(",") if tag.strip()]
                if isinstance(x, str)
                else []
            )
        )
    )

    mlb = MultiLabelBinarizer()
    tags_encoded = pd.DataFrame(
        mlb.fit_transform(features["tags"]), columns=mlb.classes_
    ).add_prefix("tag_")
    genre_encoded = pd.get_dummies(features["genre"], prefix="genre")

    processed_data = pd.concat([features, genre_encoded, tags_encoded], axis=1)
    processed_data = processed_data.drop(
        columns=["genre", "tags", "year"], errors="ignore"
    )

    processed_data = pd.DataFrame(
        scaler.fit_transform(processed_data), columns=processed_data.columns
    )

    return processed_data, metadata


# Main function to execute the pipeline
def main():
    tracks, ratings = load_data()

    # Preprocess data
    tracks, tracks_meta_data = preprocess_data(tracks)

    autoencode_recommender = AutoencodeRecommender(
        data=tracks, latent_dim=90, meta_data=tracks_meta_data
    )

    user_ids = UserProfileBuilder.get_all_users(ratings)
    test_set = user_ids[0:10]
    mean_ranking = RankingMetrics()

    for user in test_set:

        input_feature = UserProfileBuilder.aggregate_user_preference(
            user_id=user,
            ratings=ratings,
            tracks=autoencode_recommender.encoded_data,
        )
        user_ratings = UserProfileBuilder.get_rated_list(user_id=user, ratings=ratings)

        similar_indices, similar_score, similar_items = (
            autoencode_recommender.recommend_similar_items(input_feature, top_n=50)
        )

        k = 10

        # Initialize the metrics calculator
        metrics = RankingMetrics(similar_indices, user_ratings, k)
        diversity_score = DiversityMetrics.average_intra_list_distance(
            pd.DataFrame(similar_items).drop(columns=["name", "artist", "track_id"])
        )
        print(len(similar_items))
        print(f"Diversity score is: {diversity_score}")
        mean_ranking += metrics

        summary = metrics.metrics_summary()

        # Print results
        print(f"user: {user} Metrics Summary@{k}: {summary}")

    print(f"mean rating: Metrics Summary@{k}: {mean_ranking.metrics_summary()}")


if __name__ == "__main__":
    main()
