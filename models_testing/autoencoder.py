import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, layers


# Data preparation function
def prepare_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler


# Autoencoder for User and Item features (Separate Models)
class Autoencoder(Model):
    def __init__(self, latent_dimensions, data_shape):
        super(Autoencoder, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.data_shape = data_shape

        # Encoder
        self.encoder = tf.keras.Sequential(
            [
                layers.Flatten(input_shape=data_shape),
                layers.Dense(latent_dimensions, activation="relu"),
            ]
        )

        # Decoder
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(np.prod(data_shape), activation="sigmoid"),
                layers.Reshape(data_shape),
            ]
        )

    def call(self, input_data):
        encoded_data = self.encoder(input_data)
        decoded_data = self.decoder(encoded_data)
        return decoded_data


# Train function for autoencoder
def train_autoencoder(data, encoding_dim, epochs=100):
    input_dim = data.shape[1]
    autoencoder = Autoencoder(latent_dimensions=encoding_dim, data_shape=(input_dim,))
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(data, data, epochs=epochs, batch_size=32, shuffle=True)
    return autoencoder


# Function to decode the encoded data
def decode(encoded_data, autoencoder):
    # Decode the latent data to reconstruct the original data
    decoded_data = autoencoder.decoder(encoded_data)
    return decoded_data


# Function to calculate diversity scores
def calculate_diversity_scores(item_features):
    distances = pairwise_distances(item_features)
    diversity_scores = np.mean(distances, axis=1)
    return diversity_scores


# Function to recommend items with diversity
def recommend_items_with_diversity(user_id, U, V, R, diversity_weight, top_n=5):
    # Calculate base recommendations
    recommended_scores = np.dot(U[user_id], V.T)

    # Calculate diversity based on item features
    diversity_scores = calculate_diversity_scores(V)

    # Combine the scores
    final_scores = (
        1 - diversity_weight
    ) * recommended_scores + diversity_weight * diversity_scores

    # Get top N recommended items
    recommended_indices = np.argsort(final_scores)[-top_n:][::-1]
    recommended_items = recommended_indices.tolist()
    recommended_diversity = diversity_scores[recommended_indices].tolist()

    return recommended_items, recommended_diversity


# Example usage
if __name__ == "__main__":
    # Example data
    R = np.array(
        [
            [5, 4, 0, 1, 0],
            [4, 0, 0, 1, 2],
            [0, 0, 5, 4, 0],
            [1, 1, 0, 5, 0],
            [0, 0, 5, 4, 3],
        ]
    )

    # Prepare data
    data_normalized, scaler = prepare_data(R)

    # Train separate autoencoders for users and items
    encoding_dim = 3
    user_autoencoder = train_autoencoder(data_normalized, encoding_dim)

    # Use transposed data for items
    item_autoencoder = train_autoencoder(data_normalized.T, encoding_dim)

    # Extract user and item features
    U = user_autoencoder.encoder.predict(data_normalized)  # User features
    V = item_autoencoder.encoder.predict(data_normalized.T)  # Item features

    print(U)
    print(V)

    # Decode the encoded features to see reconstructions
    decoded_users = decode(U, user_autoencoder)  # Reconstruct user data
    decoded_items = decode(V, item_autoencoder)  # Reconstruct item data

    print("Decoded user data (reconstructed):")
    print(decoded_users)

    print("Decoded item data (reconstructed):")
    print(decoded_items)

    # Recommend items with diversity
    user_id_to_recommend = 0
    diversity_weight = 0.7
    recommended_items, recommended_diversity = recommend_items_with_diversity(
        user_id_to_recommend, U, V, R, diversity_weight
    )

    print("Recommended items:", recommended_items)
    print("Diversity scores for recommended items:", recommended_diversity)
