# %% [markdown]
# # Bayesian Personalized Ranking (BPR)

# %%
import sys
import cornac
import pandas as pd
import numpy as np

from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k, diversity, novelty
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

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
COL_USER = "user_id"
COL_TRACK = "track_id"
COL_COUNT = "playcount"

# %% [markdown]
# ## Load and split data
# 
# ### Load data

# %%
# Read from file
test_listening_history = pd.read_csv(header=0, delimiter="\t", filepath_or_buffer="../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt")
train_listening_history = pd.read_csv(header=0, delimiter="\t", filepath_or_buffer="../remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt")

# Reorder columns to (User, Item, Rating) - expected order for Cornac
train = train_listening_history[[COL_USER, COL_TRACK, COL_COUNT]]
test = test_listening_history[[COL_USER, COL_TRACK, COL_COUNT]]

# %% [markdown]
# ### Split data

# %%
train, test = train_listening_history, test_listening_history

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
    seed=SEED)

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
eval_map = map(test, top_k_rec, 
               col_user=COL_USER, 
               col_item=COL_TRACK, 
               col_prediction='prediction', 
               k=TOP_K,
               relevancy_method=None)

eval_ndcg = ndcg_at_k(test, top_k_rec, 
                      col_user=COL_USER, 
                      col_item=COL_TRACK, 
                      col_rating=COL_COUNT, 
                      col_prediction='prediction', 
                      k=TOP_K, 
                      relevancy_method=None)

eval_precision = precision_at_k(test, top_k_rec, 
                                col_user=COL_USER, 
                                col_item=COL_TRACK, 
                                col_prediction='prediction', 
                                k=TOP_K, 
                                relevancy_method=None)

eval_recall = recall_at_k(test, top_k_rec, 
                          col_user=COL_USER, 
                          col_item=COL_TRACK, 
                          col_prediction='prediction', 
                          k=TOP_K, 
                          relevancy_method=None)

eval_diversity = diversity(train_df=train,
                           reco_df=top_k_rec,
                           col_user=COL_USER,
                           col_item=COL_TRACK)

# Print evaluation metrics, including diversity
print("Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall,
      "NDCG:\t%f" % eval_ndcg,
      "Diversity Collab:\t%f" % eval_diversity, sep='\n')


