import os
import numpy as np
import pandas as pd
from custom_recommenders.autoencode_recommender import AutoencodeRecommender
from custom_recommenders.cosine_recommender import CosineRecommender
from custom_recommenders.softmax_recommender import SoftmaxRecommender
from metrics.diversity_metrics import DiversityMetrics
from metrics.ranking_metrics import RankingMetrics
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from utils.user_profile_builder import UserProfileBuilder
from evaluation import ContentEvaluation


# Change working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))




def load_data():
    return pd.read_csv(
        "../../remappings/data/Modified_Music_info.txt", delimiter="\t"
    ), pd.read_csv(
        "../../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt",
        delimiter="\t",
    ), pd.read_csv(
        "../../remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt",
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
    music_dataset, test_user_data, train_user_data = load_data()
    feature_data, meta_data = preprocess_data(music_dataset)


    # Initialize recommenders
    
    #autoencode_recommender = AutoencodeRecommender(data=feature_data, meta_data=meta_data, user_data=train_user_data, user_data_test=test_user_data)

    cosine_recommender = CosineRecommender(data=feature_data, meta_data=meta_data, user_data=train_user_data, user_data_test=test_user_data)

    #softmax_recommender = SoftmaxRecommender(data=feature_data, meta_data=meta_data, user_data=train_user_data)


    # Get user ids for testing
    user_ids = UserProfileBuilder.get_all_users(test_user_data)[:100]

    # prepare for ranking
    mean_ranking = RankingMetrics()
    diversity_count = 0
    mean_diversity = 0

    # Initialize empty DataFrame for all predictions
    all_prediction_data = pd.DataFrame(columns=['user', 'item', 'rank'])
   
    top_k = 25

    for i, user in enumerate(user_ids):
        print(f"Step {i + 1} out of {len(user_ids)}")
        # Get the average profile of the user tracks
        input_feature = UserProfileBuilder.aggregate_user_preference(
            user, train_user_data, feature_data
        )


        # Get the list of the users listened tracks
        user_ratings = UserProfileBuilder.get_rated_list(user, test_user_data)


        # Recommend with autoencoder
        #predicted_track_ids, predicted_track_features = autoencode_recommender.recommend_similar_items(input_feature, user_id=user, top_n=top_k)
        
        # Recommend with cosine
        predicted_track_ids, predicted_track_features = cosine_recommender.recommend_similar_items(input_feature, user_id=user, top_n=top_k)
        
        # Recommend with softmax 
        #predicted_track_ids, predicted_track_features = softmax_recommender.recommend(input_feature, user_id=user, top_n=top_k)


        prediction_data = pd.DataFrame({
            'user': [user] * len(predicted_track_ids),  
            'item': predicted_track_ids,    
            'rank': list(range(1, len(predicted_track_ids) + 1))
        })
        all_prediction_data = pd.concat([all_prediction_data, prediction_data], ignore_index=True)


        # Calculate diversity_score AILD
        diversity_score = DiversityMetrics.average_intra_list_distance(
            pd.DataFrame(predicted_track_features).drop(columns=["name", "artist", "track_id"], errors='ignore')
        )
        print(f"Diversity score: {diversity_score}")


        # Add user to the ranking metrics and display
        ranking = RankingMetrics(predicted_track_ids, user_ratings)
        mean_ranking += ranking
        diversity_count += 1
        mean_diversity += diversity_score
        print(f"user {user} Metrics Summary@10: {ranking.metrics_summary()}")


    # Print the final score
    print(f"mean rating Metrics Summary@10: {mean_ranking.metrics_summary()}")
    print(f"mean diversity score: {mean_diversity/diversity_count}")


    evaluation = ContentEvaluation()

    presicion_lk, recall_lk, hit_ratio_lk, ndgc_lk = evaluation.LenskitEvaluation(all_prediction_data)
    print (f"LENSKIT EVALUATION: precision@k: ", presicion_lk,"recall@k: ", recall_lk, "ndgc: ", ndgc_lk, "hit_ratio@k: ", hit_ratio_lk)
       
    presicion, recall, ndgc, map, diversity= evaluation.RecommenderEvaluation(all_prediction_data, top_k)
    print (f"RECOMMENDER EVALUATION: precision@k: ", presicion,"recall@k: ", recall, "ndgc: ", ndgc, "map: ", map, "diversity: ",diversity )


if __name__ == "__main__":
    main()



