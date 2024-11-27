# %% [markdown]
# # Bayesian Personalized Ranking (BPR)

# %%
import sys
import cornac
import pandas as pd
import numpy as np


from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k, diversity, novelty, serendipity, catalog_coverage, distributional_coverage
from recommenders.models.cornac.cornac_utils import predict_ranking, predict
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

#from lenskit.metrics.topn import ndcg, precision, recall

print(f"System version: {sys.version}")
print(f"Cornac version: {cornac.__version__}")

# %% [markdown]
# ## Variables

# %%
# top k items to recommend
TOP_K = 50

# Model parameters
NUM_FACTORS = 200
NUM_EPOCHS = 100

# Column names for the dataset
COL_USER = "user"
COL_TRACK = "item"
COL_COUNT = "playcount"

# %% [markdown]
# ## Load and split data
# 
# ### Load data

# %%
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


test_listening_history.columns = [COL_USER, COL_TRACK, COL_COUNT]
train_listening_history.columns = [COL_USER, COL_TRACK, COL_COUNT]

# %% [markdown]
# ### Split data

# %%
train, test = train_listening_history, test_listening_history

# Set the alpha value for the confidence transformation
alpha = 1

# Transform playcount to confidence in the training data only
train["confidence"] = 1 + alpha * np.log(1 + train[COL_COUNT])

# %% [markdown]
# ## Build a Cornac Dataset
# 
# To work with models implemented in Cornac, we need to construct an object from [Dataset](https://cornac.readthedocs.io/en/latest/data.html#module-cornac.data.dataset) class.
# 
# Dataset Class in Cornac serves as the main object that the models will interact with.  In addition to data transformations, Dataset provides a bunch of useful iterators for looping through the data, as well as supporting different negative sampling techniques.

# %%
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))

# %% [markdown]
# ## Train the BPR model

# %%
bpr = cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=0.01,
    lambda_reg=0.001,
    verbose=True,
    seed=SEED
)

# %%
with Timer() as t:
    bpr.fit(train_set)
print("Took {} seconds for training.".format(t))

# %% [markdown]
# ## Prediction and Evaluation

# %%
with Timer() as t:
    all_predictions = predict_ranking(bpr, train, usercol=COL_USER, itemcol=COL_TRACK, remove_seen=True)
print("Took {} seconds for prediction.".format(t))

all_predictions.head()

# %%
# Sort by 'user' and 'prediction' in descending order
all_prediction_sorted = all_predictions.sort_values(by=[COL_USER, 'prediction'], ascending=[True, False])

# Select the top k predictions for each user
top_k_rec = all_prediction_sorted.groupby(COL_USER).head(TOP_K)

# %%
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
# missing serendipity, catalog_coverage and distributional_coverage to be equal to the als metrics
# may be incorrect

# Print evaluation metrics, including diversity
print("Precision@K Spark:\t%f" % eval_precision,
      "Recall@K Spark:\t%f" % eval_recall,
      "NDCG Spark:\t%f" % eval_ndcg,
      "Diversity Spark:\t%f" % eval_diversity,
      "Novelty Spark:\t%f" % eval_novelty, sep='\n')

# %%
# eval_ndcg_lens = ndcg(all_predictions, test, k=TOP_K)
# eval_precision_lens = precision(all_predictions, test, k=TOP_K)
# eval_recall_lens = recall(all_predictions, test, k=TOP_K)

# # Print evaluation metrics, including diversity
# print("Precision@K Lenskit:\t%f" % eval_precision_lens,
#       "Recall@K Lenskit:\t%f" % eval_recall_lens,
#       "NDCG Lenskit:\t%f" % eval_ndcg_lens, sep='\n')


