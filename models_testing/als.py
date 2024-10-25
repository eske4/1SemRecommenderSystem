import os
import time

# spark imports
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc, col
from pyspark.sql.types import StringType, ArrayType
from pyspark.mllib.recommendation import ALS

# data science imports
import math
import numpy as np
import pandas as pd

# instantiate SparkSession object
spark = SparkSession.builder.master("local").getOrCreate()

# get spark context
sc = spark.sparkContext

# read in the dataset into pyspark DataFrame
song_ratings = spark.read.option("header", "true") \
    .option("delimiter", "\t") \
    .option("inferSchema", "true") \
    .csv("../remappings/data/Modified_Listening_History.txt")

#remapping
song_ratings = song_ratings.withColumn("track_id_temp", song_ratings.track_id).withColumn("user_id_temp", song_ratings.user_id)

song_ratings = song_ratings.withColumn("track_id", song_ratings.user_id_temp).withColumn("user_id", song_ratings.track_id_temp)

# key = old column, value = new column
mapping = {
    "track_id": "user_id",
    "user_id": "track_id",
    "playcount": "playcount"
}

song_ratings = song_ratings.select(*[col(old).alias(new) for old, new in mapping.items()])
sample = song_ratings.sample(False, 0.01, 0)

# show matrix (track, user, playcount)
song_ratings.show(2, truncate=False)

# tmp = sample.select('track_id').distinct().count()
# print('We have a total of {} distinct songs in the data sets'.format(tmp))
# tmp = sample.select('user_id').distinct().count()
# print('We have a total of {} distinct users in the data sets'.format(tmp))

# split the data into training/validation/testing sets using a 6/2/2 ratio
train, validation, test = sample.randomSplit([6.0, 2.0, 2.0], seed=99)
# cache data
train.cache()
validation.cache()
test.cache()


def train_ALS(train_data, validation_data, num_iters, reg_param, ranks):
    """
    Grid Search Function to select the best model based on RMSE of hold-out data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in reg_param:
            # train ALS model
            model = ALS.train(
                ratings=train_data,    # (userID, productID, rating) tuple
                iterations=num_iters,
                rank=rank,
                lambda_=reg,           # regularization param
                seed=99)
            # make prediction
            valid_data = validation_data.rdd.map(lambda p: (p[0], p[1]))
            predictions = model.predictAll(valid_data).map(lambda r: ((r[0], r[1]), r[2]))
            # get the rating result
            ratesAndPreds = validation_data.rdd.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
            # get the RMSE
            MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
            error = math.sqrt(MSE)
            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(rank, reg, error))
            if error < min_error:
                min_error = error
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and regularization = {}'.format(best_rank, best_regularization))
    return best_model

# hyper-param config
num_iterations = 10
ranks = [8, 10, 12, 14, 16, 18, 20]
reg_params = [0.001, 0.01, 0.05, 0.1, 0.2]

# grid search and select best model
start_time = time.time()
final_model = train_ALS(train, validation, num_iterations, reg_params, ranks)

print ('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))

