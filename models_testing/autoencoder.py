import ast
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
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
                layers.Dense(tf.math.reduce_prod(shape).numpy(), activation="sigmoid"),
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


# Function to calculate diversity
from sklearn.metrics import pairwise_distances


def calculate_diversity_scores(recommended_items):
    # Calculate pairwise distances using cosine distance
    distance_matrix = pairwise_distances(recommended_items, metric="cosine")

    # Get the upper triangle of the distance matrix (excluding the diagonal)
    triu_indices = np.triu_indices(len(distance_matrix), k=1)

    # Average distance (the lower the average, the less diverse the items are)
    diversity_score = np.mean(distance_matrix[triu_indices])

    return diversity_score


def recommend_similar_tracks(track_id, encoded_items):
    # Calculate cosine similarity between the target track and all other tracks
    sim_scores = cosine_similarity([encoded_items[track_id]], encoded_items)[0]

    # Sort by similarity and get top_n most similar tracks
    sim_track_indices = np.argsort(sim_scores)[::-1]
    sim_scores = sim_scores[sim_track_indices]

    return sim_track_indices, sim_scores

    # Define a function to make recommendations considering diversity


def recommend_with_diversity(track_id, encoded_items, diversity_weight=0.5):
    # Get similar tracks based on the encoded items
    similar_indices, sim_scores = recommend_similar_tracks(track_id, encoded_items)

    # Calculate diversity scores
    diversity_scores = calculate_diversity_scores(encoded_items[similar_indices])

    # Combine similarity and diversity scores
    combined_scores = (1 - diversity_weight) * sim_scores + diversity_weight * (
        1 - diversity_scores
    )

    # Get the final recommendations based on the combined scores
    final_recommend_indices = np.argsort(combined_scores)[::-1]

    # Return the recommended track indices and their combined scores
    return (
        similar_indices[final_recommend_indices],
        combined_scores[final_recommend_indices],
    )


# Example usage
if __name__ == "__main__":
    # Example data - replace with your actual data
    R = load_track_features("../remappings/data/Modified_Music_info.txt", 10000)

    # Prepare data
    data_normalized, scaler = prepare_data(R)

    print(np.isnan(data_normalized).any())
    print(data_normalized.shape)
    print(data_normalized)

    # Train autoencoder and get item feature matrix
    latent_dim = 32  # Adjust as needed
    input_shape = data_normalized.shape[
        1:
    ]  # Assuming data_normalized is (num_samples, num_features)
    autoencoder = Autoencoder(latent_dim, input_shape)

    autoencoder.compile(optimizer="adam", loss="mean_squared_error")

    autoencoder.fit(data_normalized, data_normalized, epochs=10)

    # Get the encoded items
    encoded_items = autoencoder.encoder.predict(data_normalized)

    # User to recommend
    diversity_weight = 0.7  # 70% diversity
    track_id_to_recommend = 388

    recommended_indices, combined_scores = recommend_with_diversity(
        track_id_to_recommend, encoded_items, diversity_weight=diversity_weight
    )

    # Output recommended track indices and their scores
    print("Recommended Track Indices:", recommended_indices)
    print("Combined Scores:", combined_scores)
