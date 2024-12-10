import cornac
import pandas as pd
import numpy as np
import warnings
import sys

from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k, diversity, novelty
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from content_based.metrics.ranking_metrics import RankingMetrics
from random import sample

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

print(f"System version: {sys.version}")
print(f"Cornac version: {cornac.__version__}")

# Constants for column names
COL_USER = "user_id"
COL_TRACK = "track_id"
COL_COUNT = "playcount"

TOP_K = 50
NUM_FACTORS = 15
NUM_EPOCHS = 50

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

# Transform playcount to binary
# train[COL_COUNT] = 1
# test[COL_COUNT] = 1

# Generate negative samples
def add_negative_samples(train, n_negative_samples):
    """
    Add N negative samples per user.
    """
    users = train[COL_USER].unique()
    items = train[COL_TRACK].unique()
    existing_interactions = set(zip(train[COL_USER], train[COL_TRACK]))

    negative_samples = []
    for user in users:
        user_items = set(train[train[COL_USER] == user][COL_TRACK])
        potential_negatives = list(set(items) - user_items)

        # Sample N negative items for the user
        if len(potential_negatives) > n_negative_samples:
            sampled_negatives = sample(potential_negatives, n_negative_samples)
        else:
            sampled_negatives = potential_negatives  # Use all if fewer than needed

        for item in sampled_negatives:
            if (user, item) not in existing_interactions:
                negative_samples.append([user, item, 0])  # Negative feedback

    negative_df = pd.DataFrame(negative_samples, columns=[COL_USER, COL_TRACK, COL_COUNT])
    return pd.concat([train, negative_df], ignore_index=True)

# Add negative samples to the training data
#train = add_negative_samples(train, N_NEGATIVE_SAMPLES)

# Create Cornac train dataset
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))

nmf = cornac.models.NMF(
        k=NUM_FACTORS,
        max_iter=NUM_EPOCHS,
        learning_rate=0.005,
        lambda_reg=0.06,
        verbose=True,
        seed=SEED,
    )

with Timer() as t:
    nmf.fit(train_set)
print("Took {} seconds for training.".format(t))

with Timer() as t:
    all_predictions = predict_ranking(nmf, train, usercol=COL_USER, itemcol=COL_TRACK, remove_seen=True)
print("Took {} seconds for prediction.".format(t))

print(all_predictions.head(20))

all_prediction_sorted = all_predictions.sort_values(by=[COL_USER, 'prediction'], ascending=[True, False])

# Select the top k predictions for each user
top_k_rec = all_prediction_sorted.groupby(COL_USER).head(TOP_K)

def statistics():
    print("\n--- Statistics on All Predictions ---")
    prediction_stats = all_predictions['prediction'].describe()
    print(prediction_stats)

    # Count of zero and non-zero predictions
    zero_predictions = (all_predictions['prediction'] == 0).sum()
    non_zero_predictions = (all_predictions['prediction'] != 0).sum()
    total_predictions = len(all_predictions)

    print(f"\nNumber of zero predictions: {zero_predictions}")
    print(f"Number of non-zero predictions: {non_zero_predictions}")
    print(f"Total predictions: {total_predictions}")

    # Percentage of zero and non-zero predictions
    percent_zero = (zero_predictions / total_predictions) * 100
    percent_non_zero = (non_zero_predictions / total_predictions) * 100

    print(f"Percentage of zero predictions: {percent_zero:.2f}%")
    print(f"Percentage of non-zero predictions: {percent_non_zero:.2f}%")

    # Optionally, check unique prediction values
    unique_predictions = all_predictions['prediction'].nunique()
    print(f"\nNumber of unique prediction values: {unique_predictions}")

    # ============================
    # Statistics on Top-K Recommendations
    # ============================

    print("\n--- Statistics on Top-K Recommendations ---")
    top_k_stats = top_k_rec['prediction'].describe()
    print(top_k_stats)

    # Count of zero and non-zero predictions in Top-K
    zero_top_k = (top_k_rec['prediction'] == 0).sum()
    non_zero_top_k = (top_k_rec['prediction'] != 0).sum()

    print(f"\nNumber of zero predictions in Top-{TOP_K}: {zero_top_k}")
    print(f"Number of non-zero predictions in Top-{TOP_K}: {non_zero_top_k}")
    print(f"Total Top-{TOP_K} predictions: {len(top_k_rec)}")

    # Percentage in Top-K
    percent_zero_top_k = (zero_top_k / len(top_k_rec)) * 100
    percent_non_zero_top_k = (non_zero_top_k / len(top_k_rec)) * 100

    print(f"Percentage of zero predictions in Top-{TOP_K}: {percent_zero_top_k:.2f}%")
    print(f"Percentage of non-zero predictions in Top-{TOP_K}: {percent_non_zero_top_k:.2f}%")

    # ============================
    # Additional Insights (Optional)
    # ============================

    # Histogram of prediction scores
    # plt.figure(figsize=(10,6))
    # sns.histplot(all_predictions['prediction'], bins=50, kde=True)
    # plt.title('Distribution of Prediction Scores')
    # plt.xlabel('Prediction Score')
    # plt.ylabel('Frequency')
    # plt.show()

    # # Histogram for Top-K predictions
    # plt.figure(figsize=(10,6))
    # sns.histplot(top_k_rec['prediction'], bins=50, kde=True, color='orange')
    # plt.title(f'Distribution of Top-{TOP_K} Prediction Scores')
    # plt.xlabel('Prediction Score')
    # plt.ylabel('Frequency')
    # plt.show()

#statistics()

ranking = RankingMetrics()



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

