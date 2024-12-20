# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# cornac imports
import cornac
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.constants import SEED

# spark imports
import sys
import pyspark
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import recommenders.evaluation.python_evaluation as py_eval

# data science imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# recommenders imports

from recommenders.utils.timer import Timer
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkDiversityEvaluation
from recommenders.utils.spark_utils import start_or_get_spark

# %%
print(f"System version: {sys.version}")
print("Spark version: {}".format(pyspark.__version__))

# %%
# top k items to recommend
TOP_K50 = 50
TOP_K25 = 25
TOP_K10 = 10

I_VALUE = 1

# Model parameters (BPR & NMF)
NUM_FACTORS = 200
NUM_EPOCHS = 100

# Column names for the dataset
COL_USER = "user_id"
COL_TRACK = "track_id"
COL_COUNT = "playcount"

# %% [markdown]
# ## Evaluation
# 
# Ranking and Diversity functions
# Usable for pySpark. To test with bpr need to use recommenders.utils.python_evaluation functions

# %%
def get_ranking_results_spark(ranking_eval):
    metrics = {
        "Precision@k": ranking_eval.precision_at_k(),
        "Recall@k": ranking_eval.recall_at_k(),
        "NDCG@k": ranking_eval.ndcg_at_k(),
        "Mean average precision": ranking_eval.map_at_k()
      
    }
    return metrics   

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
               relevancy_method=None)
      
    }
    return metrics 

def get_diversity_results_spark(diversity_eval):
    metrics = {
        "novelty": diversity_eval.novelty(), 
        "diversity": diversity_eval.diversity()
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

# %% [markdown]
# ## Create the results dataframe

# %%
cols = ["Data", "Algo", "K", "Precision@k", "Recall@k", "NDCG@k", "Mean average precision","novelty", "diversity"]
df_results = pd.DataFrame(columns=cols)

# %% [markdown]
# Summary of the evaluation

# %%
def generate_summary(data, algo, k, ranking_metrics, diversity_metrics):
    summary = {"Data": data, "Algo": algo, "K": k}

    if ranking_metrics is None:
        ranking_metrics = {           
            "Precision@k": np.nan,
            "Recall@k": np.nan,            
            "nDCG@k": np.nan,
            "MAP": np.nan,
        }
    summary.update(ranking_metrics)
    summary.update(diversity_metrics)
    return summary

# %%
# the following settings work well for debugging locally on VM - change when running on a cluster
# set up a giant single executor with many threads and specify memory cap
spark = start_or_get_spark("ALS PySpark", memory="350g", config={'spark.local.dir': "/home/manuel-albino/spark-temp", 'spark.cleaner.ttl': "true"})
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")

spark.catalog.clearCache()

# %% [markdown]
# ## Spark Loading of data

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

test_listening_history.show(2, truncate=False)
train_listening_history.show(2, truncate=False)

# %% [markdown]
# ### Train and Test data

# %%
train, test = train_listening_history, test_listening_history

# alpha = 1 

# # Transform playcount to confidence using the current alpha
# train = train.withColumn("confidence", 1 + alpha * F.log(1 + F.col(COL_COUNT))).drop(COL_COUNT)

# train.show(10, truncate=False)

print ("N train", train.cache().count())
print ("N test", test.cache().count())

# %% [markdown]
# ## ALS model creation with Confidence column

# %%
# alpha = 1 

# # Transform playcount to confidence using the current alpha
# train_with_confidence = train.withColumn("confidence", 1 + alpha * F.log(1 + F.col(COL_COUNT))).drop(COL_COUNT)

# train_with_confidence.show(10, truncate=False)

header = {
    "userCol": COL_USER,
    "itemCol": COL_TRACK,
    "ratingCol": COL_COUNT,
}

als = ALS(userCol= COL_USER, itemCol= COL_TRACK, ratingCol=COL_COUNT, rank = 10, maxIter = 40, regParam = 0.05, alpha = 60.0, coldStartStrategy="drop", nonnegative = True, implicitPrefs = True)

# %% [markdown]
# ## Training

# %%
with Timer() as train_time:
    model = als.fit(train)

print("Took {} seconds for training.".format(train_time.interval))



# %% [markdown]
# ## Prediction

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

    top_all_als = dfs_pred_exclude_train.filter(dfs_pred_exclude_train[f"train.{COL_COUNT}"].isNull()) \
        .select('pred.' + COL_USER, 'pred.' + COL_TRACK, 'pred.' + "prediction")

    # In Spark, transformations are lazy evaluation
    # Use an action to force execute and measure the test time 
    top_all_als.cache().count()

 


print("Took {} seconds for prediction.".format(test_time.interval))

top_all_als.show()


# %%


# %% [markdown]
# ### ALS metrics 

window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())    
# %%
for k in [TOP_K10]:
    # top k recommendations for each user

    top_k_reco_als = top_all_als.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= k).drop("rank")

    als_ranking_eval = SparkRankingEvaluation(
        test, 
        top_all_als, 
        k = k, 
        col_user=COL_USER, 
        col_item=COL_TRACK,
        col_rating=COL_COUNT, 
        col_prediction="prediction",
        relevancy_method="top_k"
    )

    als_ranking_metrics = get_ranking_results_spark(als_ranking_eval)

    als_diversity_eval = SparkDiversityEvaluation(
        train_df = train, 
        reco_df = top_k_reco_als,
        col_user = COL_USER, 
        col_item = COL_TRACK
    )

    als_diversity_metrics = get_diversity_results_spark(als_diversity_eval)

    als_results = generate_summary(train.count()+ test.count(), "als_wImpPref&NN", k, als_ranking_metrics, als_diversity_metrics)

    # add the models results here
    df_results.loc[I_VALUE] = als_results 
    I_VALUE+=1


als_withoutImplicit = ALS(userCol= COL_USER, itemCol= COL_TRACK, ratingCol=COL_COUNT, rank = 10, maxIter = 40, regParam = 0.05, alpha = 60.0, coldStartStrategy="drop", nonnegative = True, implicitPrefs = False)

# %% [markdown]
# ## Training

# %%
with Timer() as train_time:
    model = als_withoutImplicit.fit(train)

print("Took {} seconds for training.".format(train_time.interval))



# %% [markdown]
# ## Prediction

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

    top_all_als = dfs_pred_exclude_train.filter(dfs_pred_exclude_train[f"train.{COL_COUNT}"].isNull()) \
        .select('pred.' + COL_USER, 'pred.' + COL_TRACK, 'pred.' + "prediction")

    # In Spark, transformations are lazy evaluation
    # Use an action to force execute and measure the test time 
    top_all_als.cache().count()

print("Took {} seconds for prediction.".format(test_time.interval))

top_all_als.show()


# %%


# %% [markdown]
# ### ALS metrics 

window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())    
# %%
for k in [TOP_K10]:
    # top k recommendations for each user

    top_k_reco_als = top_all_als.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= k).drop("rank")

    als_ranking_eval = SparkRankingEvaluation(
        test, 
        top_all_als, 
        k = k, 
        col_user=COL_USER, 
        col_item=COL_TRACK,
        col_rating=COL_COUNT, 
        col_prediction="prediction",
        relevancy_method="top_k"
    )

    als_ranking_metrics = get_ranking_results_spark(als_ranking_eval)

    als_diversity_eval = SparkDiversityEvaluation(
        train_df = train, 
        reco_df = top_k_reco_als,
        col_user = COL_USER, 
        col_item = COL_TRACK
    )

    als_diversity_metrics = get_diversity_results_spark(als_diversity_eval)

    als_results = generate_summary(train.count()+ test.count(), "als_wNN", k, als_ranking_metrics, als_diversity_metrics)

    # add the models results here
    df_results.loc[I_VALUE] = als_results 
    I_VALUE+=1


als_withoutImplicit_and_NN = ALS(userCol= COL_USER, itemCol= COL_TRACK, ratingCol=COL_COUNT, rank = 10, maxIter = 40, regParam = 0.05, alpha = 60.0, coldStartStrategy="drop", nonnegative = False, implicitPrefs = False)

# %% [markdown]
# ## Training

# %%
with Timer() as train_time:
    model = als_withoutImplicit_and_NN.fit(train)

print("Took {} seconds for training.".format(train_time.interval))



# %% [markdown]
# ## Prediction

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

    top_all_als = dfs_pred_exclude_train.filter(dfs_pred_exclude_train[f"train.{COL_COUNT}"].isNull()) \
        .select('pred.' + COL_USER, 'pred.' + COL_TRACK, 'pred.' + "prediction")

    # In Spark, transformations are lazy evaluation
    # Use an action to force execute and measure the test time 
    top_all_als.cache().count()

print("Took {} seconds for prediction.".format(test_time.interval))

top_all_als.show()


# %%


# %% [markdown]
# ### ALS metrics 

window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())    
# %%
for k in [TOP_K10]:
    # top k recommendations for each user

    top_k_reco_als = top_all_als.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= k).drop("rank")

    als_ranking_eval = SparkRankingEvaluation(
        test, 
        top_all_als, 
        k = k, 
        col_user=COL_USER, 
        col_item=COL_TRACK,
        col_rating=COL_COUNT, 
        col_prediction="prediction",
        relevancy_method="top_k"
    )

    als_ranking_metrics = get_ranking_results_spark(als_ranking_eval)

    als_diversity_eval = SparkDiversityEvaluation(
        train_df = train, 
        reco_df = top_k_reco_als,
        col_user = COL_USER, 
        col_item = COL_TRACK
    )

    als_diversity_metrics = get_diversity_results_spark(als_diversity_eval)

    als_results = generate_summary(train.count()+ test.count(), "als", k, als_ranking_metrics, als_diversity_metrics)

    # add the models results here
    df_results.loc[I_VALUE] = als_results 
    I_VALUE+=1


als_withoutNN = ALS(userCol= COL_USER, itemCol= COL_TRACK, ratingCol=COL_COUNT, rank = 10, maxIter = 40, regParam = 0.05, alpha = 60.0, coldStartStrategy="drop", nonnegative = False, implicitPrefs = True)

# %% [markdown]
# ## Training

# %%
with Timer() as train_time:
    model = als_withoutNN.fit(train)

print("Took {} seconds for training.".format(train_time.interval))



# %% [markdown]
# ## Prediction

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

    top_all_als = dfs_pred_exclude_train.filter(dfs_pred_exclude_train[f"train.{COL_COUNT}"].isNull()) \
        .select('pred.' + COL_USER, 'pred.' + COL_TRACK, 'pred.' + "prediction")

    # In Spark, transformations are lazy evaluation
    # Use an action to force execute and measure the test time 
    top_all_als.cache().count()

print("Took {} seconds for prediction.".format(test_time.interval))

top_all_als.show()


# %%


# %% [markdown]
# ### ALS metrics 

window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())    
# %%
for k in [TOP_K10]:
    # top k recommendations for each user

    top_k_reco_als = top_all_als.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= k).drop("rank")

    als_ranking_eval = SparkRankingEvaluation(
        test, 
        top_all_als, 
        k = k, 
        col_user=COL_USER, 
        col_item=COL_TRACK,
        col_rating=COL_COUNT, 
        col_prediction="prediction",
        relevancy_method="top_k"
    )

    als_ranking_metrics = get_ranking_results_spark(als_ranking_eval)

    als_diversity_eval = SparkDiversityEvaluation(
        train_df = train, 
        reco_df = top_k_reco_als,
        col_user = COL_USER, 
        col_item = COL_TRACK
    )

    als_diversity_metrics = get_diversity_results_spark(als_diversity_eval)

    als_results = generate_summary(train.count()+ test.count(), "als_wImpPref", k, als_ranking_metrics, als_diversity_metrics)

    # add the models results here
    df_results.loc[I_VALUE] = als_results 
    I_VALUE+=1

# %%
print(df_results.to_string())


df_results.to_csv('../results/als_comparison.csv')

# %%
# cleanup spark instance
spark.stop()


