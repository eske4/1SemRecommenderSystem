import cornac
import pandas as pd
import numpy as np
import warnings

from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import rmse, mae, map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.utils.notebook_utils import store_metadata

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
        all_predictions = predict_ranking(nmf, train, usercol='userId', itemcol='itemId', predcol='prediction', remove_seen=True)

    print("Test columns:", test.columns)
    print("Predictions columns:", all_predictions.columns)


    print("Took {} seconds for prediction.".format(t))

    print("Predictions:")
    print(all_predictions.head())

    eval_rmse = rmse(rating_true=test, rating_pred=all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction')
    eval_mae = mae(rating_true=test, rating_pred=all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction')
    eval_map = map(test, all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction', k=k)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction', k=k)
    eval_precision = precision_at_k(test, all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction', k=k)
    eval_recall = recall_at_k(test, all_predictions, col_user='userId', col_item='itemId', col_rating='rating', col_prediction='prediction', k=k)
    
    print("RMSE:\t%f" % eval_rmse)
    print("MAE:\t%f" % eval_mae)
    print("MAP:\t%f" % eval_map)
    print("NDCG:\t%f" % eval_ndcg)
    print("Precision@K:\t%f" % eval_precision)
    print("Recall@K:\t%f" % eval_recall)

def main():
    # Configuration
    DATA_FILE_PATH = 'Modified_Listening_History.txt'  # Path to your dataset
    NROWS = 10000  # Adjust as needed
    TRAIN_RATIO = 0.8
    SEED = 42
    TOP_K = 10
    NUM_FACTORS = 20
    NUM_EPOCHS = 500


    # Step 1: Load and preprocess data
    data = read_data(DATA_FILE_PATH, nrows=NROWS)

    # Step 2: Split data into training and testing sets
    train, test = split_data(data, train_ratio=TRAIN_RATIO, seed=SEED)

    # Step 3: Convert data to Cornac's Dataset format
    train_set = create_cornac_dataset(train, seed=SEED)

    # Step 4: Train and evaluate the Cornac NMF model
    nmf_model = train_nmf_model(train_set, num_factors=NUM_FACTORS, num_epochs=NUM_EPOCHS)

    # Step 5: Evaluate Cornac NMF model
    evaluate_model(nmf_model, train, test, k=TOP_K)

if __name__ == "__main__":
    main()

# Record results for tests - ignore this cell
# store_metadata("map", eval_map)
# store_metadata("ndcg", eval_ndcg)
# store_metadata("precision", eval_precision)
# store_metadata("recall", eval_recall)