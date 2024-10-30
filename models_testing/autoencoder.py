import ast
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(
                    np.prod(shape), activation="sigmoid"
                ),  # Use np.prod instead of tf.math.reduce_prod
                layers.Reshape(shape),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_track_features(path, max_rows=None):
    df = pd.read_csv(path, delimiter="\t", nrows=max_rows)
    # Apply ast.literal_eval on each cell that contains list-like strings
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: (
                ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
            )
        )
    df.drop(columns=["tags"], errors="ignore", inplace=True)
    return df.to_numpy()


# Data preparation function
def prepare_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler


def recommend_similar_tracks(track_id, encoded_items):
    # Calculate cosine similarity between the target track and all other tracks
    sim_scores = cosine_similarity([encoded_items[track_id]], encoded_items)[0]

    # Sort by similarity and get most similar tracks
    sim_track_indices = np.argsort(sim_scores)[::-1]
    sim_scores = sim_scores[sim_track_indices]

    # Create a mask to exclude the input track from the recommendations
    mask = sim_track_indices != track_id

    # Filter out the input track from the indices and similarity scores
    filtered_indices = sim_track_indices[mask]
    filtered_scores = sim_scores[mask]

    return filtered_indices, filtered_scores


# Example usage
if __name__ == "__main__":
    # Load and prepare the data
    R = load_track_features("../remappings/data/Modified_Music_info.txt", 30000)
    data_normalized, scaler = prepare_data(R)

    # Split data into train and test sets
    train_data, test_data = train_test_split(
        data_normalized, test_size=0.2, random_state=42
    )

    # Set parameters for the autoencoder
    latent_dim = 16  # Adjust as needed
    input_shape = data_normalized.shape[
        1:
    ]  # Assuming data_normalized is (num_samples, num_features)
    autoencoder = Autoencoder(latent_dim, input_shape)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")

    # Train the autoencoder only on the train set
    autoencoder.fit(train_data, train_data, epochs=10)

    # Generate encoded representations for the entire dataset (train + test)
    encoded_train = autoencoder.encoder.predict(train_data)
    encoded_test = autoencoder.encoder.predict(test_data)

    # Select a track to recommend similar tracks for
    track_id_to_recommend = 0

    # Get similar tracks and their similarity scores in the test set
    similar_tracks, similar_tracks_scores = recommend_similar_tracks(
        track_id_to_recommend, encoded_test
    )

    # Example recommendation
    recommend_id_example = similar_tracks[2]
    print("Recommended Track Indices:", similar_tracks)
    print("Similarity score:", similar_tracks_scores)

    # Check similarity of encoded items
    print(
        "Score of recommended item similarity (encoded):",
        "Using recommended item: ",
        recommend_id_example,
        "Score is: ",
        cosine_similarity(
            [encoded_test[track_id_to_recommend]],
            [encoded_test[recommend_id_example]],
        )[0][0],
    )

    # Check similarity of original items
    print(
        "Score of recommended item similarity (original):",
        "Using recommended item: ",
        recommend_id_example,
        "Score is: ",
        cosine_similarity(
            [test_data[track_id_to_recommend]],
            [test_data[recommend_id_example]],
        )[0][0],
    )
