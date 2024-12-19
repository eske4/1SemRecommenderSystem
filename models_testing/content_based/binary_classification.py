import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate, Input
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from utils.user_profile_builder import UserProfileBuilder


# Change working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

def load_data():
    return pd.read_csv(
        "../../remappings/data/Modified_Music_info.txt", delimiter="\t"
    ), pd.read_csv(
        "../../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt",
        delimiter="\t",
    ), pd.read_csv(
        "../../remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt",
        delimiter="\t",
    )

def preprocess_data(data, scaler=MinMaxScaler()):
    meta = data[["name", "artist", "track_id"]]
    features = data.drop(columns=["name", "artist", "track_id"], errors="ignore")
    features["genre"] = features["genre"].astype("category")


    # Handle tags, ensuring that we don't attempt to split non-string values
    features["tags"] = features["tags"].apply(
        lambda x: (
            eval(x)
            if isinstance(x, str) and x.startswith("[")
            else ([tag.strip() for tag in x.split(",")] if isinstance(x, str) else [])
        )
    )


    # Initialize MultiLabelBinarizer and fit it to the tags data
    mlb = MultiLabelBinarizer()
    tags_encoded = pd.DataFrame(
        mlb.fit_transform(features["tags"]), columns=mlb.classes_
    ).add_prefix("tag_")


    genre_encoded = pd.get_dummies(features["genre"], prefix="genre")


    # Combine the processed data
    processed_data = pd.concat([features, genre_encoded, tags_encoded], axis=1)
    processed_data = processed_data.drop(
        columns=["genre", "tags", "year"], errors="ignore"
    )


    # Scale the features
    processed_data = pd.DataFrame(
        scaler.fit_transform(processed_data), columns=processed_data.columns
    )


    return processed_data, meta



# Load and process the data
music_dataset, test_user_data, train_user_data = load_data()
train_track_ids = train_user_data['track_id'].unique()
train_music_dataset = music_dataset[music_dataset['track_id'].isin(train_track_ids)].reset_index(drop=True)
all_feature_data, meta_data = preprocess_data(music_dataset)
feature_data, meta_data = preprocess_data(train_music_dataset)

user_ids = UserProfileBuilder.get_all_users(test_user_data)[:1000]
profile_builer = UserProfileBuilder()

user_preferences_train = []

for user in user_ids:
    # Compute the input feature for the current user
    input_feature = UserProfileBuilder.aggregate_user_preference(
        user, train_user_data, all_feature_data
    )
    # Append the input feature (as a Series or dict) to the list
    user_preferences_train.append(input_feature)


train_data = pd.read_csv("../../remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt", delimiter="\t")
unique_tracks_train = train_data['track_id'].unique()
# Binarize the 'heard' column: 1 if heard, 0 if not
train_data['heard'] = (train_data['playcount'] > 0).astype(int)
# Create the interaction matrix: rows are users, columns are unique tracks, values are 1 or 0
interaction_matrix = train_data.pivot_table(index='user_id', columns='track_id', values='heard', fill_value=0)


interaction_matrix_subset = interaction_matrix.iloc[:1000, :]

#shape [25893 rows, 128 columns]
song_features_df = feature_data
print("song features: ", song_features_df)
#shape [23795 rows, 128 columns]
user_preferences_df = pd.DataFrame(user_preferences_train)
print("user_preferences_df: ", user_preferences_df)
#shape [23795 row, 25893 coloumns]
interaction_matrix_df = pd.DataFrame(interaction_matrix_subset)
print("interaction_matrix_df: ", interaction_matrix_df)


# Song features (e.g., 5 features per song: [tempo, genre, mood, popularity, duration])
song_features = song_features_df.to_numpy()
# User preferences (e.g., averaged song features they liked)
user_preferences = user_preferences_df.to_numpy()
# Labels: Did the user like the song? (1 = yes, 0 = no)
labels = interaction_matrix_df.to_numpy()

# Build the model
# Input: User preferences and song features concatenated
user_input = Input(shape=(user_preferences.shape[1],), name="user_input")  # User preference vector
song_input = Input(shape=(song_features.shape[1],), name="song_input")    # Song feature vector

# Model__________________________________________________________

# Concatenate user and song features
concat = tf.keras.layers.Concatenate()([user_input, song_input])
# Neural network layers
x = Dense(32, activation='relu')(concat)
x = Dense(16, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Output layer: probability of liking the song
# Model definition
model = tf.keras.models.Model(inputs=[user_input, song_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare the data for training
# Flatten user-song pairs for training
user_feature_matrix = np.repeat(user_preferences, song_features.shape[0], axis=0)
song_feature_matrix = np.tile(song_features, (user_preferences.shape[0], 1))
flattened_labels = labels.flatten()

# Train the model
model.fit([user_feature_matrix, song_feature_matrix], flattened_labels, epochs=1, batch_size=64)

# Predict: 
user_1_features = np.tile(user_preferences[0], (song_features.shape[0], 1))  
predictions = model.predict([user_1_features, song_features])  
print("Predictions for User 1 (probabilities):", predictions)
