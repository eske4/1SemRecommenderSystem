import ast

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


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


def load_csv(path, max_rows=None):
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
def calculate_diversity_scores(recommended_scores):
    # Implement diversity calculation logic here
    diversity_scores = np.random.rand(
        len(recommended_scores)
    )  # Example: random diversity scores
    return diversity_scores


# Function to recommend items with diversity
def recommend_items_with_diversity(user_id, U, V, R, diversity_weight, top_n=5):
    # Calculate recommendations based on user and item matrices
    recommended_scores = np.dot(U[user_id], V.T)

    # Calculate diversity scores
    diversity_scores = calculate_diversity_scores(recommended_scores)

    # Combine scores based on diversity weight
    final_scores = (
        1 - diversity_weight
    ) * recommended_scores + diversity_weight * diversity_scores

    # Get top recommended items
    recommended_indices = np.argsort(final_scores)[-top_n:][::-1]  # Get top_n items
    recommended_items = recommended_indices.tolist()
    recommended_diversity = diversity_scores[
        recommended_indices
    ].tolist()  # Get diversity scores for recommended items

    return recommended_items, recommended_diversity


# Example usage
if __name__ == "__main__":
    # Example data - replace with your actual data
    R = load_csv("../remappings/data/Modified_Music_info.txt", 10000)

    # Prepare data
    data_normalized, scaler = prepare_data(R)

    # Train autoencoder and get item feature matrix
    latent_dim = 32  # You can adjust this value as needed
    autoencoder = Autoencoder(latent_dim, data_normalized.shape[1:])

    autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())
    autoencoder.fit(data_normalized, data_normalized, epochs=100)

    # User to recommend
    user_id_to_recommend = 0
    diversity_weight = 0.7  # 70% diversity
