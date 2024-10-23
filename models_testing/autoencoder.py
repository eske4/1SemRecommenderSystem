import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def load_csv(path, max_rows=None):
    return np.genfromtxt(path, delimiter=",", skip_header=1, max_rows=max_rows)


# Data preparation function
def prepare_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler


# Autoencoder model definition
def build_autoencoder(input_dim, encoding_dim):
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
    decoded = tf.keras.layers.Dense(input_dim, activation="sigmoid")(encoded)

    autoencoder = tf.keras.models.Model(input_layer, decoded)
    encoder = tf.keras.models.Model(input_layer, encoded)
    return autoencoder, encoder


# Function to train the autoencoder
def train_autoencoder(data, encoding_dim, epochs=100):
    input_dim = data.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(data, data, epochs=epochs, batch_size=32, shuffle=True)
    return encoder


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
    R = load_csv("data/Music Info.csv", 10000)

    # Prepare data
    data_normalized, scaler = prepare_data(R)

    # Train autoencoder and get user feature matrix
    encoding_dim = 32  # You can adjust this value as needed
    encoder = train_autoencoder(data_normalized, encoding_dim)
    U = encoder.predict(data_normalized)  # User features
    V = encoder.predict(data_normalized.T)  # Item features

    # User to recommend
    user_id_to_recommend = 0
    diversity_weight = 0.7  # 70% diversity

    # Recommend items
    recommended_items, recommended_diversity = recommend_items_with_diversity(
        user_id_to_recommend, U, V, R, diversity_weight
    )

    print("Recommended items:", recommended_items)
    print("Diversity scores for recommended items:", recommended_diversity)
