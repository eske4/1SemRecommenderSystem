import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------------------------------

#                                            data handling

# paths to datafiles

file_path_music_data = r"remappings/data/Modified_Music_info.txt"
file_path_user_data = r"remappings/data/Modified_Listening_History.txt"

# read data files (.txt with \t seperation)
music_data = pd.read_csv(file_path_music_data, delimiter="\t")
user_data = pd.read_csv(file_path_user_data, delimiter="\t")

# merge user and music data
merged_data = pd.merge(music_data, user_data, on="track_id")

# adjust data amount
subset_dataset = merged_data.head(100000)

# -----------------------------------------------------------------------------------------------------

#                                          data preprocessing

# select which features the model should use
features = subset_dataset[["danceability", "energy", "valence", "tempo", "loudness"]]

# select what the model should predict
target = subset_dataset["name"]

# data splitting
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------------------------------------------------------------------------

#                                               models

# ---------------------------multiclass prediction (softmax activationfunction)------------------------

# number of unique track IDs
num_unique_tracks = target.max() + 1

# model layers
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_unique_tracks, activation="softmax"),
    ]
)

# compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# -----------------------------------------------------------------------------------------------------

#                                          model training

# training parameters
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# example user parameters for array of songs 
user_input = np.array([[0.355, 0.918, 0.24, 148.114, -4.36],[0.355, 0.918, 0.24, 148.114, -4.36],[0.355, 0.918, 0.24, 148.114, -4.36],[0.355, 0.918, 0.24, 148.114, -4.36],[0.355, 0.918, 0.24, 148.114, -4.36]])
user_input_scaled = scaler.transform(user_input)

# number of songs wanted to be recommended to user
num_recommendations = 10

# -----------------------------------------------------------------------------------------------------

#                                           predictions

# ---------------------------multiclass prediction (softmax activationfunction)------------------------

# get predictions for user input
predictions = model.predict(user_input_scaled)

# get top n predictions
top_n_indices = np.argsort(predictions[0])[-num_recommendations:][::-1]

# print predictions
recommended_track_ids = [music_data["track_id"].values[i] for i in top_n_indices]
print("Recommended Track IDs:", recommended_track_ids)

#-------------------------------------------------------------------------------------------------------

#                                            get songs



track_id_file_path = r"remappings\data\names.txt"
track_id = pd.read_csv(track_id_file_path, delimiter="\t")
track_id.columns = ['track_name', 'track_id']

recommended_track_names = track_id[track_id['track_id'].isin(recommended_track_ids)]

print("Recommended Track names:", recommended_track_names)


