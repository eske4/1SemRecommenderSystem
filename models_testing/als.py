# %%
import os
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# spark imports
import sys
import pyspark
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# data science imports
import numpy as np
import pandas as pd

# recommenders imports
from recommenders.utils.timer import Timer
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkDiversityEvaluation
from recommenders.utils.spark_utils import start_or_get_spark

from tqdm import tqdm

# %%
print(f"System version: {sys.version}")
print("Spark version: {}".format(pyspark.__version__))

# %% [markdown]
# ## Variables

# %%
# top k items to recommend
TOP_K = 50

# Column names for the dataset
COL_USER = "user_id"
COL_TRACK = "track_id"
COL_COUNT = "playcount"

# %% [markdown]
# ## Spark Init

# %%
# the following settings work well for debugging locally on VM - change when running on a cluster
# set up a giant single executor with many threads and specify memory cap
spark = start_or_get_spark("ALS PySpark", memory="16g", config={'spark.local.dir': "/home/matildeschade/spark-temp", 'spark.cleaner.ttl': "true"})
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")

spark.catalog.clearCache()

# %% [markdown]
# ## Load and Split Data

# %% [markdown]
# ### Load Data

# %%
# Read in the dataset into pyspark DataFrame    
test_listening_history = spark.read.option("header", "true") \
    .option("delimiter", "\t") \
    .option("inferSchema", "true") \
    .csv("../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt")
    
train_listening_history = spark.read.option("header", "true") \
    .option("delimiter", "\t") \
    .option("inferSchema", "true") \
    .csv("../remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt")

# Change columns to correct place (user_id, track_id, playcount)
test_listening_history = test_listening_history.withColumn("track_id_temp", test_listening_history.track_id).withColumn("user_id_temp", test_listening_history.user_id)
test_listening_history = test_listening_history.withColumn("track_id", test_listening_history.user_id_temp).withColumn("user_id", test_listening_history.track_id_temp)

train_listening_history = train_listening_history.withColumn("track_id_temp", train_listening_history.track_id).withColumn("user_id_temp", train_listening_history.user_id)
train_listening_history = train_listening_history.withColumn("track_id", train_listening_history.user_id_temp).withColumn("user_id", train_listening_history.track_id_temp)

# key = old column, value = new column
mapping = {
    "track_id": COL_USER,
    "user_id": COL_TRACK,
    "playcount": COL_COUNT
}

test_listening_history = test_listening_history.select(*[F.col(old).alias(new) for old, new in mapping.items()])
train_listening_history = train_listening_history.select(*[F.col(old).alias(new) for old, new in mapping.items()])

# Sample
test_listening_history = test_listening_history.sample(False, 0.005, 0)
train_listening_history = train_listening_history.sample(False, 0.005, 0)

test_listening_history.show(2, truncate=False)
train_listening_history.show(2, truncate=False)

# %% [markdown]
# ### Split Data

# %%
train, test = train_listening_history, test_listening_history

# alpha = 1 

# # Transform playcount to confidence using the current alpha
# train = train.withColumn("confidence", 1 + alpha * F.log(1 + F.col(COL_COUNT))).drop(COL_COUNT)

# train.show(10, truncate=False)

print ("N train", train.cache().count())
print ("N test", test.cache().count())

# %% [markdown]
# ## Train de ALS model

# %% [markdown]
# ### Specify ALS hyperparameters

# %%
ranks = [10, 20, 30, 40]
maxIters = [10, 20, 30, 40]
regParams = [.05, .1, .15]
alphas = [20, 40, 60, 80]

# %%
# For loop will automatically create and store ALS models
model_list = []

for r in tqdm(ranks):
    for mi in maxIters:
        for rp in regParams:
            for a in alphas:
                model_list.append(ALS(userCol= COL_USER, itemCol= COL_TRACK, ratingCol=COL_COUNT, rank = r, maxIter = mi, regParam = rp, alpha = a, coldStartStrategy="drop", nonnegative = True, implicitPrefs = True))

# Print the model list, and the length of model_list
print (model_list, "Length of model_list: ", len(model_list))

# Validate
len(model_list) == (len(ranks)*len(maxIters)*len(regParams)*len(alphas))

# %%
# Expected percentile rank error metric function
def ROEM(predictions, userCol = COL_USER, itemCol = COL_TRACK, ratingCol = COL_COUNT):
  # Creates table that can be queried
  predictions.createOrReplaceTempView("predictions")
  
  # Sum of total number of plays of all songs
  denominator = predictions.groupBy().sum(ratingCol).collect()[0][0]    

  # Calculating rankings of songs predictions by user
  spark.sql("SELECT " + userCol + " , " + ratingCol + " , PERCENT_RANK() OVER (PARTITION BY " + userCol + " ORDER BY prediction DESC) AS rank FROM predictions").createOrReplaceTempView("rankings")

  # Multiplies the rank of each song by the number of plays and adds the products together
  numerator = spark.sql('SELECT SUM(' + ratingCol + ' * rank) FROM rankings').collect()[0][0]
  performance = numerator/denominator
  
  return performance

# %%
# Building 5 folds within the training set.
train1, train2, train3, train4, train5 = train.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed = 1)
fold1 = train2.union(train3).union(train4).union(train5)
fold2 = train3.union(train4).union(train5).union(train1)
fold3 = train4.union(train5).union(train1).union(train2)
fold4 = train5.union(train1).union(train2).union(train3)
fold5 = train1.union(train2).union(train3).union(train4)

foldlist = [(fold1, train1), (fold2, train2), (fold3, train3), (fold4, train4), (fold5, train5)]

# Empty list to fill with ROEMs from each model
ROEMS = []

# Loops through all models and all folds
for model in model_list:
    for ft_pair in foldlist:

        # Fits model to fold within training data
        fitted_model = model.fit(ft_pair[0])

        # Generates predictions using fitted_model on respective CV test data
        predictions = fitted_model.transform(ft_pair[1])

        # Generates and prints a ROEM metric CV test data
        r = ROEM(predictions)
        print ("ROEM: ", r)

    # Fits model to all of training data and generates preds for test data
    v_fitted_model = model.fit(train)
    v_predictions = v_fitted_model.transform(test)
    v_ROEM = ROEM(v_predictions)

    # Adds validation ROEM to ROEM list
    ROEMS.append(v_ROEM)
    print ("Validation ROEM: ", v_ROEM)

# %%
# Extract the best_model
best_model = model_list[38]

# Extract the Rank
best_rank = best_model.getRank()
print ("Rank: ", best_rank)

# Extract the MaxIter value
best_maxIter = best_model.getMaxIter()
print ("MaxIter: ", best_maxIter)

# Extract the RegParam value
best_regParam = best_model.getRegParam()
print ("RegParam: ", best_regParam)

# Extract the Alpha value
best_alpha = best_model.getAlpha()
print ("Alpha: ", best_alpha)

# %%
# header = {
#     "userCol": COL_USER,
#     "itemCol": COL_TRACK,
#     "ratingCol": 'confidence',
# }

# als = ALS(
#     rank=best_rank,
#     maxIter=best_maxIter,
#     implicitPrefs=True,
#     regParam=best_regParam,
#     alpha=best_alpha,
#     coldStartStrategy='drop',
#     nonnegative=True,
#     seed=42,
#     **header)

# %%
with Timer() as train_time:
    model = best_model.fit(train)

print("Took {} seconds for training.".format(train_time.interval))

# %% [markdown]
# ## Predict and Evaluate

# %% [markdown]
# ### Prediction

# %%
with Timer() as test_time:

    # Get the cross join of all user-item pairs and score them.
    users = train.select(COL_USER).distinct()
    items = train.select(COL_TRACK).distinct()
    user_item = users.crossJoin(items)
    dfs_pred = model.transform(user_item)

    # Remove seen items.
    dfs_pred_exclude_train = dfs_pred.alias("pred").join(
        train.alias("train"),
        (dfs_pred[COL_USER] == train[COL_USER]) & (dfs_pred[COL_TRACK] == train[COL_TRACK]),
        how='outer'
    )

    top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train[f"train.{COL_COUNT}"].isNull()) \
        .select('pred.' + COL_USER, 'pred.' + COL_TRACK, 'pred.' + "prediction")

    # In Spark, transformations are lazy evaluation
    # Use an action to force execute and measure the test time 
    top_all.cache().count()

print("Took {} seconds for prediction.".format(test_time.interval))

# %%
top_all.show()

# %%
# top_k_rec = model.recommendForAllUsers(TOP_K)
# top_k_rec.show(10)

# %%
# with Timer() as t:
#     predictions = model.transform(test)
# print("Took {} seconds for prediction.".format(t))

# predictions.cache().show()

# %% [markdown]
# ### Evaluation

# %%
rank_eval = SparkRankingEvaluation(test, top_all, k = TOP_K, col_user=COL_USER, col_item=COL_TRACK, 
                                    col_rating=COL_COUNT, col_prediction="prediction", relevancy_method="top_k")

# %%
print("Model:\tALS",
      "Precision@K:\t%f" % rank_eval.precision_at_k(),
      "Recall@K:\t%f" % rank_eval.recall_at_k(), 
      "NDCG:\t%f" % rank_eval.ndcg_at_k(),
      "MAP:\t%f" % rank_eval.map_at_k(), sep='\n')

# %%
# Cleanup spark instance
spark.stop()


