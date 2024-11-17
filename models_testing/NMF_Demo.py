import cornac
import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm

from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import rmse, mae, map, ndcg_at_k, precision_at_k, recall_at_k, diversity, novelty
from recommenders.models.cornac.cornac_utils import predict_ranking, predict
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.utils.notebook_utils import store_metadata
from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants for column names
COL_USER = "user_id"
COL_TRACK = "track_id"
COL_COUNT = "playcount"
COL_PREDICTION = "prediction"

# Configuration
DATA_FILE_PATH = 'remappings/data/Modified_Listening_History.txt'  # Path to your dataset
TRACK_IDS_PATH = "remappings/data/track_ids.txt" # Path to original trackId mapping dataset
MUSIC_INFO_PATH = "data/Music Info.csv" # Path to music info with orifinal trackIds
NROWS = 10000 # Adjust as needed
TRAIN_RATIO = 0.8
ALPHA = 1 # Confidence level for training data
SEED = 42
TOP_K = 20
NUM_FACTORS = 20
NUM_EPOCHS = 200
USER_ID = 0 # User Id for which to generate playlist

def read_data():
    """
    Reads and preprocesses the user-song interaction data.

    Returns:
        pd.DataFrame: Preprocesses user-song interaction data.
    """
    data = pd.read_csv(
        DATA_FILE_PATH,
        sep='\t',
        dtype={COL_TRACK: int, COL_USER: int, COL_COUNT: int},
        nrows=NROWS  # Adjust or remove nrows to read more data
    )

    # Reorder columns to (User, Item, Rating) - expected order for Cornac
    data = data[[COL_USER, COL_TRACK, COL_COUNT]]

    # Inspect the first few rows to ensure correct reading
    print("Sampled Data:")
    print(data.head())

    # # Binarizes or sets confidence level for the play counts
    # # data['rating'] = 1
    # alpha = 1
    # data[COL_COUNT] = 1 + alpha * np.log1p(data[COL_COUNT])  # log1p(x) = log(1 + x)

    num_users = data[COL_USER].nunique()
    num_tracks = data[COL_TRACK].nunique()

    print(f"Dataset Size: {len(data)} interactions")
    print(f"Number of users: {num_users}")
    print(f"Number of tracks: {num_tracks}")

    return data

def split_data(data):
    """
    Splits the data into training and testing sets

    Args:
        data (pd.DataFrame): The complete dataset.

    Returns:
        train_data (pd.DataFrame): Training set.
        test_data (pd.DataFrame): Testing set.
    """
    train, test = python_random_split(data, ratio=TRAIN_RATIO, seed=SEED)
    print(f"Training data: {len(train)} interactions")
    print(f"Test data: {len(test)} interactions")
    

    # Transform playcount to confidence in the training data only
    train[COL_COUNT] = 1 + ALPHA * np.log1p(train[COL_COUNT])

    return train, test

def create_cornac_dataset(train):
    """
    Converts pandas DataFrames into Cornac's Dataset format.
    
    Args:
        train (pd.DataFrame): Training set.
    
    Returns:
        train_set (cornac.data.Dataset): Cornac-formatted training set.
    """

    # Create Cornac train dataset
    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

    # Calculate sparsity and density
    num_users = train_set.num_users
    num_items = train_set.num_items
    num_observed_interactions = len(train)

    total_possible_interactions = num_users * num_items
    num_zero_interactions = total_possible_interactions - num_observed_interactions

    sparsity = num_zero_interactions / total_possible_interactions
    density = num_observed_interactions / total_possible_interactions

    print("\nInteraction Matrix Statistics:")
    print('Number of users: {}'.format(train_set.num_users))
    print('Number of items: {}'.format(train_set.num_items))
    print(f"Total possible interactions: {total_possible_interactions}")
    print(f"Number of observed interactions: {num_observed_interactions}")
    print(f"Number of zero interactions: {num_zero_interactions}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Density: {density:.4f}")

    return train_set

def train_nmf_model(train_set, seed=None):
    nmf = cornac.models.NMF(
        k=NUM_FACTORS,
        max_iter=NUM_EPOCHS,
        learning_rate=0.01,
        lambda_u=0.06,
        lambda_v=0.06,
        lambda_bu=0.02,
        lambda_bi=0.02,
        use_bias=False,
        verbose=True,
        seed=seed,
    )

    with Timer() as t:
        nmf.fit(train_set)
    print("Took {} seconds for training.".format(t))

    return nmf

def evaluate_model(nmf, train, test):
    
    with Timer() as t:
        all_predictions = predict_ranking(nmf, train, usercol=COL_USER, itemcol=COL_TRACK, predcol=COL_PREDICTION, remove_seen=True)
        #all_predictions = predict(nmf, train, usercol=COL_USER, itemcol=COL_TRACK, predcol=COL_PREDICTION, remove_seen=True)

    print("Took {} seconds for prediction.".format(t))

    print("Predictions:")
    print(all_predictions.head())

    # Sort predictions for evaluation
    all_prediction_sorted = all_predictions.sort_values(by=[COL_USER, COL_PREDICTION], ascending=[True, False])

    # Select the top k predictions for diversity and novelty
    top_k_rec = all_prediction_sorted.groupby(COL_USER).head(TOP_K).reset_index(drop=True)

    eval_rmse = rmse(rating_true=test, rating_pred=all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION)
    eval_mae = mae(rating_true=test, rating_pred=all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION)
    eval_map = map(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION, k=TOP_K)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION, k=TOP_K)
    eval_precision = precision_at_k(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_prediction=COL_PREDICTION, k=TOP_K)
    eval_recall = recall_at_k(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_prediction=COL_PREDICTION, k=TOP_K)
    eval_diversity = diversity(train_df=train, reco_df=top_k_rec, col_user=COL_USER, col_item=COL_TRACK)
    eval_novelty = novelty(train_df=train, reco_df=top_k_rec, col_user=COL_USER, col_item=COL_TRACK)


    print("RMSE:\t%f" % eval_rmse)
    print("MAE:\t%f" % eval_mae)
    print("MAP:\t%f" % eval_map)
    print("NDCG:\t%f" % eval_ndcg)
    print("Precision@K:\t%f" % eval_precision)
    print("Recall@K:\t%f" % eval_recall)
    print("Diversity:\t%f" % eval_diversity)
    print("Novelty:\t%f" % eval_novelty)

    return all_predictions

def generate_playlist_for_user(nmf, data):
    """
    Generates and prints the top k playlist for a given user using nmf score method.

    Parameters:
    - nmf (Model): Trained NMF model.
    - data (pd.DataFrame): DataFrame containing user-item interactions with columns ['userId', 'itemId', 'rating'].
    """

    # Load the track_map and music_info_df
    track_map = {
        int(remapped_id): original_id for original_id, remapped_id in
        (line.strip().split() for line in open(TRACK_IDS_PATH))
    }
    music_info_df = pd.read_csv(MUSIC_INFO_PATH)

    # Map external user ID to internal user index
    user_id_map = nmf.train_set.uid_map
    if USER_ID in user_id_map:
        user_idx = user_id_map[USER_ID]
    else:
        print(f"User ID {USER_ID} not found in the training data.")
        return

    # Get scores for all items for the user
    user_scores = nmf.score(user_idx=user_idx)

    # Map internal track indices back to external track IDs
    reverse_track_id_map = {idx: track_id for track_id, idx in nmf.train_set.iid_map.items()}

    # Get tracks the user has already interacted with
    seen_tracks = set(data[data[COL_USER] == USER_ID][COL_TRACK])

    # Filter out seen tracks and sort scores
    unseen_tracks = [
        (track_idx, score, track_id)
        for track_idx, score in enumerate(user_scores)
        if (track_id := reverse_track_id_map.get(track_idx)) and track_id not in seen_tracks
    ]

    # Sort the unseen tracks by score in descending order and pick the top k
    top_tracks = sorted(unseen_tracks, key=lambda x: x[1], reverse=True)[:TOP_K]

    # Map to original track IDs and get song details
    playlist = [
        (
            music_info_df.loc[music_info_df[COL_TRACK] == track_map[track_id], "name"].values[0],
            music_info_df.loc[music_info_df[COL_TRACK] == track_map[track_id], "artist"].values[0],
            prediction
        )
        for _, prediction, track_id in top_tracks
    ]

    # Print the playlist
    print(f"\nTop {TOP_K} playlist for user {USER_ID}:")
    for i, (song, artist, prediction) in enumerate(playlist, 1):
        print(f"{i}. {song} by {artist} ({prediction:.2f})")

def main():
    # Step 1: Load and preprocess data
    data = read_data()

    # Step 2: Split data into training and testing sets
    train, test = split_data(data)

    # Step 3: Convert data to Cornac's Dataset format
    train_set = create_cornac_dataset(train)

    # Step 4: Train and evaluate the Cornac NMF model
    nmf_model = train_nmf_model(train_set, SEED)

    # Step 5: Evaluate Cornac NMF model
    all_predictions = evaluate_model(nmf_model, train, test)

    # Step 6: Generate playlist
    generate_playlist_for_user(nmf_model, data)

if __name__ == "__main__":
    main()