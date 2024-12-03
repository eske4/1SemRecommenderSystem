import cornac
import pandas as pd
import numpy as np
import warnings
import sys
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k, diversity, novelty
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

# Set the random seed for reproducibility
np.random.seed(SEED)

# Get all unique tracks from the training data
all_tracks = set(train[COL_TRACK].unique())

# Create a dictionary mapping each user to the set of tracks they've interacted with
user_seen_tracks = train.groupby(COL_USER)[COL_TRACK].apply(set).to_dict()

recommendations = []

for user, seen_tracks in user_seen_tracks.items():
    # Get the set of unseen tracks for the user
    unseen_tracks = list(all_tracks - seen_tracks)
    
    # Determine the number of items to sample
    num_to_sample = TOP_K

    # Randomly sample unseen tracks
    sampled_tracks = np.random.choice(unseen_tracks, size=num_to_sample, replace=False)
    
    # Assign random prediction scores to the sampled tracks
    prediction_scores = np.random.rand(num_to_sample)
    
    # Create recommendation entries with user, track, and prediction score
    user_recommendations = pd.DataFrame({
        COL_USER: user,
        COL_TRACK: sampled_tracks,
        'prediction': prediction_scores
    })
    
    # Sort by prediction score in descending order
    user_recommendations = user_recommendations.sort_values(by='prediction', ascending=False)
    
    # Append to the list of recommendations
    recommendations.append(user_recommendations)

# Concatenate all user recommendations into a single DataFrame
top_k_rec = pd.concat(recommendations, ignore_index=True)
# Evaluation Metrics
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