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

def read_data(file_path, nrows=None):
    """
    Reads and preprocesses the user-song interaction data.

    Args:
        file_path (str): Path to the dataset file.
        nrows (int, optional): Number of rows to read. Defaults to None (reads all).

    Returns:
        pd.DataFrame: Preprocesses user-song interaction data.
    """
    data = pd.read_csv(
        file_path,
        sep='\t',
        dtype={'track_id': int, 'user_id': int, 'playcount': int},
        nrows=nrows  # Adjust or remove nrows to read more data
    )

    # Rename columns to match expected format
    data.rename(columns={"track_id": "itemId", "user_id": "userId", "playcount": "rating"}, inplace=True)

    # Inspect the first few rows to ensure correct reading
    print("Sampled Data:")
    print(data.head())

    # Binarizes or sets confidence level for the play counts
    # data['rating'] = 1
    alpha = 1
    data['rating'] = 1 + alpha * np.log1p(data['rating'])  # log1p(x) = log(1 + x)

    num_users = data['userId'].nunique()
    num_tracks = data['itemId'].nunique()

    print(f"Dataset Size: {len(data)} interactions")
    print(f"Number of users: {num_users}")
    print(f"Number of tracks: {num_tracks}")

    return data

def split_data(data, train_ratio, seed):
    """
    Splits the data into training and testing sets using Recommenders' framwork

    Args:
        data (pd.DataFrame): The complete dataset.

    Returns:
        train_data (pd.DataFrame): Training set.
        test_data (pd.DataFrame): Testing set.
    """
    train, test = python_random_split(data, ratio=train_ratio, seed=seed)
    print(f"Training data: {len(train)} interactions")
    print(f"Test data: {len(test)} interactions")
    test = test[['userId', 'itemId', 'rating']]
    return train, test

def create_cornac_dataset(train, seed):
    """
    Converts pandas DataFrames into Cornac's Dataset format using itertuples.
    
    Args:
        train_data (pd.DataFrame): Training set.
    
    Returns:
        train_set (cornac.data.Dataset): Cornac-formatted training set.
    """
    # Reorder columns to (userId, itemId, rating) - expected order for Cornac
    train = train[['userId', 'itemId', 'rating']]

    # Create Cornac train dataset
    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=seed)

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

def train_nmf_model(train_set, num_factors, num_epochs, seed=None):
    nmf = cornac.models.NMF(
        k=num_factors,
        max_iter=num_epochs,
        learning_rate=0.005,
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

def evaluate_model(nmf, train, test, k):

    with Timer() as t:
        all_predictions = predict_ranking(nmf, train, usercol='userId', itemcol='itemId', predcol='prediction', remove_seen = True)
        #all_predictions = predict(nmf, test, usercol='userId', itemcol='itemId', predcol='prediction')


    print("Test columns:", test.columns)
    print("Predictions columns:", all_predictions.columns)


    print("Took {} seconds for prediction.".format(t))

    print("Predictions:")
    print(all_predictions.head())

    # Limit the predictions to the top k items per user for diversity and novelty calculations
    all_predictions = all_predictions.sort_values(by=['userId', 'prediction'], ascending=[True, False])
    all_predictions = all_predictions.groupby('userId').head(k).reset_index(drop=True)

    eval_rmse = rmse(rating_true=test, rating_pred=all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction')
    eval_mae = mae(rating_true=test, rating_pred=all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction')
    eval_map = map(test, all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction', k=k)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction', k=k)
    eval_precision = precision_at_k(test, all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction', k=k)
    eval_recall = recall_at_k(test, all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction', k=k)
    eval_diversity = diversity( train_df=train, reco_df=all_predictions, col_user='userId', col_item='itemId')
    eval_novelty = novelty(train_df=train, reco_df=all_predictions, col_user='userId', col_item='itemId')

    print("RMSE:\t%f" % eval_rmse)
    print("MAE:\t%f" % eval_mae)
    print("MAP:\t%f" % eval_map)
    print("NDCG:\t%f" % eval_ndcg)
    print("Precision@K:\t%f" % eval_precision)
    print("Recall@K:\t%f" % eval_recall)
    print("Diversity:\t%f" % eval_diversity)
    print("Novelty:\t%f" % eval_novelty)

    return all_predictions

def generate_playlist_for_user(user_id, all_predictions, k):
    """
    Generates and prints the top k playlist for a given user.

    Parameters:
    - user_id (int): The ID of the user.
    - all_predictions (pd.DataFrame): DataFrame containing prediction results with columns ['userId', 'itemId', 'prediction'].
    - k (int): Number of top items to include in the playlist.
    """

    track_ids_path = "remappings/data/track_ids.txt"
    music_info_path = "data/Music Info.csv"

    # Load the track_map and music_info_df with the defined paths
    track_map = {int(remapped_id): original_id for original_id, remapped_id in 
                (line.strip().split() for line in open(track_ids_path))}
    music_info_df = pd.read_csv(music_info_path)

    # Step 2: Filter the top k predictions for the specified user
    user_top_k = all_predictions[all_predictions['userId'] == user_id].nlargest(k, 'prediction')

    # Step 3: Map remapped itemIds to original track_ids and fetch song details
    playlist = [
        (music_info_df.loc[music_info_df['track_id'] == track_map[row['itemId']], 'name'].values[0],
        music_info_df.loc[music_info_df['track_id'] == track_map[row['itemId']], 'artist'].values[0])
        for _, row in user_top_k.iterrows() if row['itemId'] in track_map
    ]

    # Step 4: Print the playlist
    print(f"\nTop {k} playlist for user {user_id}:")
    for i, (song, artist) in enumerate(playlist, 1):
        print(f"{i}. {song} by {artist}")

def main():
    # Configuration
    DATA_FILE_PATH = 'remappings/data/Modified_Listening_History.txt'  # Path to your dataset
    NROWS = 20000 # Adjust as needed
    TRAIN_RATIO = 0.8
    SEED = 42
    TOP_K = 10
    NUM_FACTORS = 20
    NUM_EPOCHS = 100


    # Step 1: Load and preprocess data
    data = read_data(DATA_FILE_PATH, nrows=NROWS)

    # Step 2: Split data into training and testing sets
    train, test = split_data(data, train_ratio=TRAIN_RATIO, seed=SEED)

    # Step 3: Convert data to Cornac's Dataset format
    train_set = create_cornac_dataset(train, seed=SEED)

    # Step 4: Train and evaluate the Cornac NMF model
    nmf_model = train_nmf_model(train_set, num_factors=NUM_FACTORS, num_epochs=NUM_EPOCHS, seed=SEED)

    # Step 5: Evaluate Cornac NMF model
    all_predictions = evaluate_model(nmf_model, train, test, k=TOP_K)

    generate_playlist_for_user(1, all_predictions, TOP_K)

if __name__ == "__main__":
    main()

# Record results for tests - ignore this cell
# store_metadata("map", eval_map)
# store_metadata("ndcg", eval_ndcg)
# store_metadata("precision", eval_precision)
# store_metadata("recall", eval_recall)