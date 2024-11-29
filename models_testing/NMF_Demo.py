import cornac
import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm

from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import rmse, mae, map, ndcg_at_k, precision_at_k, recall_at_k, diversity, novelty
from recommenders.models.cornac.cornac_utils import predict_ranking, predict,predict_ranking_topk
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from cornac.eval_methods import RatioSplit
from cornac.metrics import MAE, RMSE, Recall, Precision, NDCG, MAP
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants for column names
COL_USER = "user_id"
COL_TRACK = "track_id"
COL_COUNT = "playcount"
COL_PREDICTION = "prediction"

# Configuration
# DATA_FILE_PATH = 'data/User Listening History.csv'  # Path to your dataset
# DATA_FILE_PATH = 'remappings/data/Modified_Listening_History.txt'  # Path to your dataset
DATA_FILE_PATH = '1SemRecommenderSystem/remappings/data/dataset/Full.txt'  # Path to your dataset
TRAIN_SET_PATH = '1SemRecommenderSystem/remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt'
TEST_SET_PATH = '1SemRecommenderSystem/remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt'
TRACK_IDS_PATH = "remappings/data/track_ids.txt" # Path to original trackId mapping dataset
MUSIC_INFO_PATH = "data/Music Info.csv" # Path to music info with orifinal trackIds
NROWS = 50000 # Adjust as needed
TRAIN_RATIO = 0.8
ALPHA = 1 # Confidence level for training data
SEED = 42
TOP_K = 50
NUM_FACTORS = 500
NUM_EPOCHS = 200
USER_ID = 2000 # User Id for which to generate playlist

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

    user_song_list = data.groupby(COL_USER, observed=True)[[COL_TRACK, COL_COUNT]].apply(lambda x: list(zip(x[COL_TRACK], x[COL_COUNT]))).to_dict()

    user_song_list = {user: songs for user, songs in user_song_list.items() if len(songs) >= 50}

    data = data[data[COL_USER].isin(user_song_list.keys())] 

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

def split_data(data=None):
    """
    Splits the data into training and testing sets

    Args:
        data (pd.DataFrame): The complete dataset.

    Returns:
        train_data (pd.DataFrame): Training set.
        test_data (pd.DataFrame): Testing set.
    """
    # train, test = python_random_split(data, ratio=TRAIN_RATIO, seed=SEED)
    train = pd.read_csv(
        TRAIN_SET_PATH,
        sep='\t',
        dtype={COL_TRACK: int, COL_USER: int, COL_COUNT: int},
        nrows=NROWS
    )

    # Read the testing dataset
    test = pd.read_csv(
        TEST_SET_PATH,
        sep='\t',
        dtype={COL_TRACK: int, COL_USER: int, COL_COUNT: int},
        nrows=NROWS
    )

    train = train[[COL_USER, COL_TRACK, COL_COUNT]]
    test = test[[COL_USER, COL_TRACK, COL_COUNT]]

    print(f"Training data: {len(train)} interactions")
    print(f"Test data: {len(test)} interactions")

    # Calculate unique users in each dataset
    unique_train_users = set(train[COL_USER].unique())
    unique_test_users = set(test[COL_USER].unique())

    # Calculate statistics
    num_unique_train_users = len(unique_train_users)
    num_unique_test_users = len(unique_test_users)
    common_users = unique_train_users.intersection(unique_test_users)
    num_common_users = len(common_users)

    # Print statistics
    print(f"Number of unique users in the training dataset: {num_unique_train_users}")
    print(f"Number of unique users in the testing dataset: {num_unique_test_users}")
    print(f"Number of users common to both datasets: {num_common_users}")

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

    # nmf.rank()
    return nmf

def train_nmf_model_via_cornac(data, seed=None):

    # Step 2: Prepare data for Cornac
    data_tuples = list(zip(data[COL_USER], data[COL_TRACK], data[COL_COUNT]))

    # Step 3: Define the evaluation method
    eval_method = RatioSplit(
        data=data_tuples,
        test_size=1 - TRAIN_RATIO,
        exclude_unknowns=False,
        verbose=True,
        seed=SEED,
    )

    # Step 4: Define the NMF model
    nmf = cornac.models.NMF(
        k=NUM_FACTORS,
        max_iter=NUM_EPOCHS,
        learning_rate=0.005,
        lambda_u=0.06,
        lambda_v=0.06,
        use_bias=False,
        verbose=True,
        seed=seed,
    )

    # Step 5: Define the evaluation metrics
    mae = MAE()
    rmse = RMSE()
    rec_k = Recall(k=TOP_K)
    pre_k = Precision(k=TOP_K)
    ndcg_k = NDCG(k=TOP_K)
    map_k = MAP()

    # Step 6: Run the experiment
    cornac.Experiment(
        eval_method=eval_method,
        models=[nmf],
        metrics=[mae, rmse, rec_k, pre_k, ndcg_k, map_k],
        user_based=True,
    ).run()

def evaluate_model(nmf, train, test):
    
    with Timer() as t:
        all_predictions = predict_ranking(nmf, train, usercol=COL_USER, itemcol=COL_TRACK, predcol=COL_PREDICTION, remove_seen=True)
        #all_predictions = predict(nmf, train, usercol=COL_USER, itemcol=COL_TRACK, predcol=COL_PREDICTION, remove_seen=True)
        # all_predictions = predict_ranking_topk(
        #     model=nmf,
        #     data=train,  # The data containing users and items you want predictions for
        #     usercol=COL_USER,
        #     itemcol=COL_TRACK,
        #     predcol=COL_PREDICTION,
        #     remove_seen=True,       # Set to True to exclude seen items
        #     top_k=TOP_K             # Number of top items per user
        # )
    print("Took {} seconds for prediction.".format(t))

    print("Predictions:")
    print(all_predictions.head())

    # Calculate statistics for the predictions
    unique_predicted_users = all_predictions[COL_USER].nunique()

    # Print statistics
    print(f"Number of unique users in all_predictions: {unique_predicted_users}")
    # Sort predictions for evaluation
    all_prediction_sorted = all_predictions.sort_values(by=[COL_USER, COL_PREDICTION], ascending=[True, False])

    # Select the top k predictions for diversity and novelty
    top_k_rec = all_prediction_sorted.groupby(COL_USER).head(TOP_K).reset_index(drop=True)

    # eval_rmse = rmse(rating_true=test, rating_pred=all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION)
    # eval_mae = mae(rating_true=test, rating_pred=all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION)
    eval_map = map(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION, k=TOP_K)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION, k=TOP_K)
    eval_precision = precision_at_k(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_prediction=COL_PREDICTION, k=TOP_K)
    eval_recall = recall_at_k(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_prediction=COL_PREDICTION, k=TOP_K)
    eval_diversity = diversity(train_df=train, reco_df=top_k_rec, col_user=COL_USER, col_item=COL_TRACK)
    eval_novelty = novelty(train_df=train, reco_df=top_k_rec, col_user=COL_USER, col_item=COL_TRACK)


    # print("RMSE:\t%f" % eval_rmse)
    # print("MAE:\t%f" % eval_mae)
    print("MAP:\t%f" % eval_map)
    print("NDCG:\t%f" % eval_ndcg)
    print("Precision@K:\t%f" % eval_precision)
    print("Recall@K:\t%f" % eval_recall)
    print("Diversity:\t%f" % eval_diversity)
    print("Novelty:\t%f" % eval_novelty)

    return all_predictions

def recommend_songs_nmf(user_id, model, n_recommendations=10):
    """
    Generates song recommendations for a user based on NMF factors.
    """
    # Get the internal user index
    user_id_map = model.train_set.uid_map
    if user_id not in user_id_map:
        print(f"User {user_id} not found in the training set.")
        return []
    
    user_idx = user_id_map[user_id]

    # Compute predictions for all items
    user_factors = model.u_factors
    item_factors = model.i_factors
    predictions = np.dot(user_factors[user_idx, :], item_factors.T)

    # Sort items by prediction scores and get top indices
    top_indices = np.argsort(predictions)[::-1][:n_recommendations]

    # Map back to external item IDs
    item_id_map = model.train_set.iid_map
    index_to_item_id = {index: item_id for item_id, index in item_id_map.items()}
    top_items = [index_to_item_id[idx] for idx in top_indices]

    return top_items

def evaluate_model_nmf(user_test_data, user_train_data, model, n_recommendations=10):
    """
    Evaluates the NMF model using precision and recall with a progress bar.
    """
    precisions = []
    recalls = []

    # Convert test and train data into dictionaries for efficient lookup
    user_test_data = (
        user_test_data.groupby(COL_USER)[COL_TRACK]
        .apply(list)
        .to_dict()
    )

    user_train_data = (
        user_train_data.groupby(COL_USER)[COL_TRACK]
        .apply(list)
        .to_dict()
    )

    # Use tqdm to show progress while iterating over users
    for user, true_tracks in tqdm(user_test_data.items(), desc="Evaluating users"):
        # Check if the user is in the train data
        if user in user_train_data:
            # Get top recommendations for the user
            recommended_tracks = recommend_songs_nmf(
                user,
                model,
                n_recommendations=n_recommendations,
            )

            # Calculate precision and recall
            true_positives = len(set(recommended_tracks) & set(true_tracks))
            precision = true_positives / len(recommended_tracks) if recommended_tracks else 0
            recall = true_positives / len(true_tracks) if true_tracks else 0

            precisions.append(precision)
            recalls.append(recall)

    # Compute average precision and recall
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    return avg_precision, avg_recall

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

def random_recommender(train_df, top_k=10):
    """
    Generates random recommendations for each user in train_df.

    Args:
        train_df (pd.DataFrame): Training data containing user-item interactions.
        top_k (int): Number of recommendations per user.

    Returns:
        pd.DataFrame: Randomly recommended items for each user with random scores.
    """
    COL_USER = 'user_id'
    COL_TRACK = 'track_id'
    COL_PREDICTION = 'prediction'

    # Get unique users and items
    users = train_df[COL_USER].unique()
    items = train_df[COL_TRACK].unique()
    all_items = set(items)
    
    # Build a mapping from user to items seen
    user_items_train = train_df.groupby(COL_USER)[COL_TRACK].apply(set).to_dict()
    
    recommendations = []
    
    for user in tqdm(users, desc="Generating random recommendations"):
        seen_items = user_items_train.get(user, set())
        unseen_items = np.array(list(all_items - seen_items))
        if len(unseen_items) == 0:
            continue  # No unseen items to recommend
        num_to_sample = min(top_k, len(unseen_items))
        sampled_items = np.random.choice(unseen_items, size=num_to_sample, replace=False)
        scores = np.random.rand(num_to_sample)
        user_recs = pd.DataFrame({
            COL_USER: [user]*num_to_sample,
            COL_TRACK: sampled_items,
            COL_PREDICTION: scores
        })
        recommendations.append(user_recs)
        
    pred_df = pd.concat(recommendations, ignore_index=True)
    return pred_df

def evaluate_random_recommender(train_df, test_df, top_k=10):
    """
    Evaluates the random recommender using standard metrics.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Testing data.
        top_k (int): Number of recommendations per user.

    Returns:
        None
    """
    COL_USER = 'user_id'
    COL_TRACK = 'track_id'
    COL_COUNT = 'playcount'
    COL_PREDICTION = 'prediction'

    # Generate recommendations
    pred_df = random_recommender(train_df, top_k=top_k)
    
    # Evaluate the recommendations
    eval_map = map(
        test_df, pred_df,
        col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION, k=top_k
    )
    eval_ndcg = ndcg_at_k(
        test_df, pred_df,
        col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction=COL_PREDICTION, k=top_k
    )
    eval_precision = precision_at_k(
        test_df, pred_df,
        col_user=COL_USER, col_item=COL_TRACK, col_prediction=COL_PREDICTION, k=top_k
    )
    eval_recall = recall_at_k(
        test_df, pred_df,
        col_user=COL_USER, col_item=COL_TRACK, col_prediction=COL_PREDICTION, k=top_k
    )
    eval_diversity = diversity(
        train_df=train_df, reco_df=pred_df,
        col_user=COL_USER, col_item=COL_TRACK
    )
    eval_novelty = novelty(
        train_df=train_df, reco_df=pred_df,
        col_user=COL_USER, col_item=COL_TRACK
    )

    print("MAP:\t%f" % eval_map)
    print("NDCG:\t%f" % eval_ndcg)
    print("Precision@K:\t%f" % eval_precision)
    print("Recall@K:\t%f" % eval_recall)
    print("Diversity:\t%f" % eval_diversity)
    print("Novelty:\t%f" % eval_novelty)

def main():
    # Step 1: Load and preprocess data
    data = read_data()

    # Step 2: Split data into training and testing sets
    train, test = split_data()

    # Step 3: Convert data to Cornac's Dataset format
    train_set = create_cornac_dataset(train)

    # Step 4: Train and evaluate the Cornac NMF model
    #nmf_model = train_nmf_model(train_set, SEED)

    train_nmf_model_via_cornac(data, SEED)
    # Step 5: Evaluate Cornac NMF model
    #all_predictions = evaluate_model(nmf_model, train, test)
    #evaluate_model_nmf(test, train, nmf_model, TOP_K)

    # Step 6: Generate playlist
    #generate_playlist_for_user(nmf_model, data)

    # Generate all possible user-item pairs

    # Evaluate the random recommender
    #evaluate_random_recommender(train, test, top_k=TOP_K)
if __name__ == "__main__":
    main()