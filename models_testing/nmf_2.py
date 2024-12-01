import cornac
import pandas as pd
import numpy as np
import warnings
import sys

from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k, diversity, novelty
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

print(f"System version: {sys.version}")
print(f"Cornac version: {cornac.__version__}")

# Constants for column names
COL_USER = "user_id"
COL_TRACK = "track_id"
COL_COUNT = "playcount"

TOP_K = 50
NUM_FACTORS = 200
NUM_EPOCHS = 100

# Read from file
test = pd.read_csv(header=0, 
                   delimiter="\t", 
                   filepath_or_buffer="../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt",
                   dtype={COL_TRACK: int, COL_USER: int, COL_COUNT: int},
                   nrows=20000
                   )
train = pd.read_csv(header=0, 
                    delimiter="\t", 
                    filepath_or_buffer="../remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt",
                    dtype={COL_TRACK: int, COL_USER: int, COL_COUNT: int},
                    nrows=20000
                    )

# Reorder columns to (User, Item, Rating) - expected order for Cornac
train = train[[COL_USER, COL_TRACK, COL_COUNT]]
test = test[[COL_USER, COL_TRACK, COL_COUNT]]

# Transform playcount to confidence in the training data only
# alpha = 1
# train[COL_COUNT] = 1 + alpha * np.log(1 + train[COL_COUNT])

# Transform playcount to binary
train[COL_COUNT] = 1
test[COL_COUNT] = 1

# Create Cornac train dataset
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

with Timer() as t:
    eval_map = map(
        test,
        top_k_rec,
        col_user=COL_USER,
        col_item=COL_TRACK,
        col_prediction='prediction',
        k=TOP_K,
        relevancy_method=None
    )
print("Took {} seconds for MAP calculation.".format(t))

# NDCG
with Timer() as t:
    eval_ndcg = ndcg_at_k(
        test,
        top_k_rec,
        col_user=COL_USER,
        col_item=COL_TRACK,
        col_rating=COL_COUNT,
        col_prediction='prediction',
        k=TOP_K,
        relevancy_method=None
    )
print("Took {} seconds for NDCG calculation.".format(t))

# Precision@K
with Timer() as t:
    eval_precision = precision_at_k(
        test,
        top_k_rec,
        col_user=COL_USER,
        col_item=COL_TRACK,
        col_prediction='prediction',
        k=TOP_K,
        relevancy_method=None
    )
print("Took {} seconds for Precision@K calculation.".format(t))

# Recall@K
with Timer() as t:
    eval_recall = recall_at_k(
        test,
        top_k_rec,
        col_user=COL_USER,
        col_item=COL_TRACK,
        col_prediction='prediction',
        k=TOP_K,
        relevancy_method=None
    )
print("Took {} seconds for Recall@K calculation.".format(t))

# Diversity
with Timer() as t:
    eval_diversity = diversity(
        train_df=train,
        reco_df=top_k_rec,
        col_user=COL_USER,
        col_item=COL_TRACK
    )
print("Took {} seconds for Diversity calculation.".format(t))

# Novelty
with Timer() as t:
    eval_novelty = novelty(
        train_df=train,
        reco_df=top_k_rec,
        col_user=COL_USER,
        col_item=COL_TRACK
    )
print("Took {} seconds for Novelty calculation.".format(t))

# Print evaluation metrics, including diversity
print("Map Spark:\t%f" % eval_map,
      "Precision@K Spark:\t%f" % eval_precision,
      "Recall@K Spark:\t%f" % eval_recall,
      "NDCG Spark:\t%f" % eval_ndcg,
      "Diversity Spark:\t%f" % eval_diversity,
      "Novelty Spark:\t%f" % eval_novelty, sep='\n')