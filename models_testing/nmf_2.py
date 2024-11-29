import cornac
import pandas as pd
import numpy as np
import warnings
import sys
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

print(f"System version: {sys.version}")
print(f"Cornac version: {cornac.__version__}")

# Constants for column names
COL_USER = "user_id"
COL_TRACK = "track_id"
COL_COUNT = "playcount"

ALPHA = 1 # Confidence level for training data
SEED = 42
TOP_K = 50
NUM_FACTORS = 200
NUM_EPOCHS = 100

# Read from file
test_listening_history = pd.read_csv(header=0, delimiter="\t", filepath_or_buffer="../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt")
train_listening_history = pd.read_csv(header=0, delimiter="\t", filepath_or_buffer="../remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt")

# Change columns to correct place (user_id, track_id, playcount)
track_test = test_listening_history["track_id"]
user_test = test_listening_history["user_id"]

track_train = train_listening_history["track_id"]
user_train = train_listening_history["user_id"]

test_listening_history["track_id"] = user_test
test_listening_history["user_id"] = track_test

train_listening_history["track_id"] = user_train
train_listening_history["user_id"] = track_train

train, test = train_listening_history, test_listening_history

# Set the alpha value for the confidence transformation
alpha = 1

# Transform playcount to confidence in the training data only
train["confidence"] = 1 + alpha * np.log(1 + train[COL_COUNT])

train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))

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
    seed=SEED,
)

with Timer() as t:
    nmf.fit(train_set)
print("Took {} seconds for training.".format(t))

with Timer() as t:
    all_predictions = predict_ranking(nmf, train, usercol=COL_USER, itemcol=COL_TRACK, remove_seen=True)
print("Took {} seconds for prediction.".format(t))

all_predictions.head()

all_prediction_sorted = all_predictions.sort_values(by=[COL_USER, 'prediction'], ascending=[True, False])

# Select the top k predictions for each user
top_k_rec = all_prediction_sorted.groupby(COL_USER).head(TOP_K)

eval_map = map(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_rating=COL_COUNT, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test, all_predictions, col_user=COL_USER, col_item=COL_TRACK, col_prediction='prediction', k=TOP_K)
eval_diversity = diversity(
    train_df=train,
    reco_df=top_k_rec,
    col_user=COL_USER,
    col_item=COL_TRACK
)
eval_novelty = novelty(
    train_df=train,
    reco_df=top_k_rec,
    col_user=COL_USER,
    col_item=COL_TRACK
)

# Print evaluation metrics, including diversity
print("Precision@K Spark:\t%f" % eval_precision,
      "Recall@K Spark:\t%f" % eval_recall,
      "NDCG Spark:\t%f" % eval_ndcg,
      "Diversity Spark:\t%f" % eval_diversity,
      "Novelty Spark:\t%f" % eval_novelty, sep='\n')