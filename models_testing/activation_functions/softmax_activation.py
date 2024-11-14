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


#for content based dont need userdata 
# merge user and music data
merged_data = pd.merge(music_data, user_data, on="track_id")

# -----------------------------------------------------------------------------------------------------

#                                          data preprocessing

# select which features the model should use
selected_features = ["danceability", "energy","key", "loudness", "mode", "speechiness","acousticness", "instrumentalness", "liveness", "valence", "tempo", ]
features = music_data[selected_features]

# select what the model should predict
target = music_data["name"]

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
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.4)

#find track parameters based on song name id 
user_history = [0] #insert name id for user id
user_history_all_parameters = music_data[music_data["name"].isin(user_history)]
user_history_selected_parameters = user_history_all_parameters[selected_features]
user_history_selected_parameters_array = user_history_selected_parameters.values.tolist()

# example user parameters for user song history 
user_input = np.array(user_history_selected_parameters_array)
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
recommended_track_ids = [music_data["track_id"].values[i] for i in top_n_indices]

# get song name
name_id_file_path = r"remappings\data\names.txt"
name_id = pd.read_csv(name_id_file_path, delimiter="\t")
name_id.columns = ['track_name', 'id']
recommended_track_names = name_id[name_id['id'].isin(recommended_track_ids)]

print("Recommended Tracks:",recommended_track_names)

# get similarity score



#brug dette til similarity score
'''
def recommend_similar_tracks(track_id, encoded_data, items):
    sim_scores = cosine_similarity([encoded_data[track_id]], encoded_data)[0]
    sorted_indices = np.argsort(sim_scores)[::-1]
'''
