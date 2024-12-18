# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# cornac imports
import cornac
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.constants import SEED

import recommenders.evaluation.python_evaluation as py_eval

# data science imports
import numpy as np
import pandas as pd
from random import sample
import random 

from recommenders.utils.timer import Timer

# top k items to recommend
TOP_K50 = 50
TOP_K25 = 25
TOP_K10 = 10

# negative samples
NEGATIVE_SAMPLES_10 = 10
NEGATIVE_SAMPLES_25 = 25
NEGATIVE_SAMPLES_50 = 50

I_VALUE = 1

# Model parameters (BPR)
NUM_FACTORS = 200
NUM_EPOCHS = 100

# Column names for the dataset
COL_USER = "user_id"
COL_TRACK = "track_id"
COL_COUNT = "playcount"

def calculate_hit_at_k(test, top_k_rec, user_col=COL_USER, item_col=COL_TRACK):
    """
    Calculate Hit@K for top-K recommendations.

    Args:
        top_k_rec (pd.DataFrame): DataFrame containing top-K recommendations with columns [user_col, item_col].
        test_set (pd.DataFrame): DataFrame containing the ground truth with columns [user_col, item_col].
        user_col (str): Column name for user IDs.
        item_col (str): Column name for item IDs.

    Returns:
        float: Hit@K value.
    """
    # Create a dictionary mapping users to their relevant items
    test_dict = test.groupby(user_col)[item_col].apply(set).to_dict()

    # Check hits
    hits = sum(
        any(item in test_dict[user] for item in group[item_col].values)
        for user, group in top_k_rec.groupby(user_col)
        if user in test_dict
    )

    # Total number of users in the test set
    total_users = len(test_dict)

    # Calculate Hit@K
    hit_rate = hits / total_users

    return hit_rate

def get_ranking_results_python(test, top_k_rec_bpr, k):
    metrics = {
        "Precision@k": py_eval.precision_at_k(test, top_k_rec_bpr, 
                                col_user=COL_USER, 
                                col_item=COL_TRACK, 
                                col_prediction='prediction', 
                                k=k, 
                                relevancy_method=None),
        "Recall@k": py_eval.recall_at_k(test, top_k_rec_bpr, 
                          col_user=COL_USER, 
                          col_item=COL_TRACK, 
                          col_prediction='prediction', 
                          k=k, 
                          relevancy_method=None),
        "Hit@k": calculate_hit_at_k(test, top_k_rec_bpr),
        "NDCG@k": py_eval.ndcg_at_k(test, top_k_rec_bpr, 
                      col_user=COL_USER, 
                      col_item=COL_TRACK, 
                      col_rating=COL_COUNT, 
                      col_prediction='prediction', 
                      k=k, 
                      relevancy_method=None),
        "Mean average precision": py_eval.map(test, top_k_rec_bpr, 
               col_user=COL_USER, 
               col_item=COL_TRACK, 
               col_prediction='prediction', 
               k=k,
               relevancy_method=None),
                }
    return metrics 

def get_diversity_results_python(train, top_k_rec_bpr):
    metrics = {
        "novelty": py_eval.novelty(train_df=train,
                       reco_df=top_k_rec_bpr,
                       col_user=COL_USER,
                       col_item=COL_TRACK), 
        "diversity": py_eval.diversity(train_df=train,
                           reco_df=top_k_rec_bpr,
                           col_user=COL_USER,
                           col_item=COL_TRACK)
    }
    return metrics 

# Generate negative samples
def add_negative_samples(train, n_negative_samples):
    """
    Add N negative samples per user.
    """
    random.seed(SEED)

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

# ## Create the results dataframe

cols = ["Algo", "Neg Samples", "K", "Precision@k", "Recall@k", "Hit@k", "NDCG@k", "Mean average precision","novelty", "diversity"]
df_results = pd.DataFrame(columns=cols)

# %% [markdown]
# Summary of the evaluation

# %%
def generate_summary(algo, data, k, ranking_metrics, diversity_metrics):
    summary = {"Algo": algo, "Neg Samples": data, "K": k}

    if ranking_metrics is None:
        ranking_metrics = {           
            "Precision@k": np.nan,
            "Recall@k": np.nan,
            "Hit@k": np.nan,            
            "nDCG@k": np.nan,
            "MAP": np.nan,
        }
    summary.update(ranking_metrics)
    summary.update(diversity_metrics)
    return summary

# %% [markdown]
# ## Pandas Loading of data

# %%
# Read from file
test_listening_history = pd.read_csv(header=0, 
                                     delimiter="\t", 
                                     filepath_or_buffer="../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt",
                                     nrows=None)
train_listening_history = pd.read_csv(header=0, 
                                      delimiter="\t", 
                                      filepath_or_buffer="../remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt",
                                      nrows=None)

train, test = train_listening_history, test_listening_history

# Reorder columns to (User, Item, Rating) - expected order for Cornac
train = train[[COL_USER, COL_TRACK, COL_COUNT]]
test = test[[COL_USER, COL_TRACK, COL_COUNT]]

# Transform playcount to binary
train[COL_COUNT] = 1

# Define the negative sample sizes
negative_sample_sizes = [NEGATIVE_SAMPLES_10, NEGATIVE_SAMPLES_25, NEGATIVE_SAMPLES_50]

# Loop through each negative sample size
for n_negative_samples in negative_sample_sizes:
    # Add negative samples to the train set
    train_with_negatives = add_negative_samples(train, n_negative_samples)

    # Create Cornac Dataset
    train_set = cornac.data.Dataset.from_uir(train_with_negatives.itertuples(index=False), seed=SEED)

    print(f"Training NMF model with {n_negative_samples} negative samples...")

    # Train the NMF model
    nmf = cornac.models.NMF(
        k=NUM_FACTORS,
        max_iter=NUM_EPOCHS,
        learning_rate=0.01,
        lambda_reg=0.001,
        verbose=True,
        seed=SEED,
    )

    with Timer() as t:
        nmf.fit(train_set)
    print(f"Took {t} seconds for training with {n_negative_samples} negative samples.")

    # Predict rankings
    with Timer() as t:
        all_predictions_nmf = predict_ranking(nmf, train_with_negatives, usercol=COL_USER, itemcol=COL_TRACK, remove_seen=True)
    print(f"Took {t} seconds for prediction with {n_negative_samples} negative samples.")

    # Sort predictions
    all_prediction_sorted_nmf = all_predictions_nmf.sort_values(by=[COL_USER, 'prediction'], ascending=[True, False])

    # Evaluate for each TOP_K
    for k in [TOP_K10, TOP_K25, TOP_K50]:
        # Select the top k predictions for each user
        top_k_rec_nmf = all_prediction_sorted_nmf.groupby(COL_USER).head(k)

        # Calculate ranking and diversity metrics
        nmf_ranking_metrics = get_ranking_results_python(test, top_k_rec_nmf, k)
        nmf_diversity_metrics = get_diversity_results_python(train_with_negatives, top_k_rec_nmf)

        # Generate results summary
        nmf_results = generate_summary(
            algo="NMF",
            data=f"{n_negative_samples}",
            k=k,
            ranking_metrics=nmf_ranking_metrics,
            diversity_metrics=nmf_diversity_metrics,
        )

        # Add results to the dataframe
        df_results.loc[I_VALUE] = nmf_results
        I_VALUE += 1

# Save final results to CSV
print(df_results.to_string())
df_results.to_csv('../results/nmf_negative_sampling_comparison.csv')