import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, layers


# Define Autoencoder model class
class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        # Encoder part: compress the data into a latent representation
        self.encoder = tf.keras.Sequential(
            [
                layers.Flatten(),  # Flatten the user-item matrix
                layers.Dense(latent_dim, activation="relu"),  # Latent dimension
            ]
        )
        # Decoder part: reconstruct the data from the latent representation
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(
                    np.prod(shape), activation="sigmoid"
                ),  # Reconstruct the matrix
                layers.Reshape(shape),  # Reshape back to original shape
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_ratings_data(path, max_rows=None):
    ratings = pd.read_csv(path, delimiter="\t", nrows=max_rows)

    # Check the column names to ensure they match expected
    print("Column names in the dataset:", ratings.columns)

    # Strip whitespace from columns
    ratings.columns = ratings.columns.str.strip()

    # Optionally, rename columns if needed
    if "track_id" in ratings.columns:
        ratings = ratings.rename(columns={"track_id": "item_id"})

    return ratings


def create_user_item_matrix(ratings):
    # Ensure 'item_id' and 'user_id' columns exist before pivoting
    if "item_id" not in ratings.columns or "user_id" not in ratings.columns:
        raise ValueError("'item_id' or 'user_id' column missing in the dataset")

    # Pivot the data to create the user-item matrix
    user_item_matrix = ratings.pivot(
        index="user_id", columns="item_id", values="playcount"
    ).fillna(0)
    return user_item_matrix


# Normalize data
def normalize_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler


# Train the autoencoder on the user-item matrix
def train_autoencoder(train_data, latent_dim):
    input_shape = train_data.shape[1:]
    autoencoder = Autoencoder(latent_dim, input_shape)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(train_data, train_data, epochs=20, batch_size=16, shuffle=True)
    return autoencoder


from sklearn.metrics.pairwise import cosine_similarity


def recommend_items_with_similarity(autoencoder, data, user_ids, item_ids, top_n=5):
    reconstructed = autoencoder.predict(data)
    similarity_matrix = cosine_similarity(reconstructed.T)

    # Create a dictionary to map item IDs to indices
    item_idx_map = {item: idx for idx, item in enumerate(item_ids)}

    recommendations = {}
    for user_idx, user_id in enumerate(user_ids):
        user_scores = reconstructed[user_idx]
        sorted_items = sorted(zip(item_ids, user_scores), key=lambda x: -x[1])

        # Exclude interacted items
        interacted_items = set(
            [
                item
                for item, score in sorted_items
                if data[user_idx][item_idx_map[item]] > 0
            ]
        )
        recommended_items = [
            item for item, _ in sorted_items if item not in interacted_items
        ][:top_n]

        if len(recommended_items) < top_n:
            # Add similar items
            similar_items = []
            for item in recommended_items:
                item_idx = item_idx_map[item]
                similar_items += [
                    item
                    for item, _ in sorted(
                        zip(item_ids, similarity_matrix[item_idx]), key=lambda x: -x[1]
                    )
                    if item not in interacted_items
                ]
                if len(similar_items) >= top_n - len(recommended_items):
                    break
            recommended_items += similar_items[: top_n - len(recommended_items)]

        recommendations[user_id] = recommended_items

    return recommendations


# Generate recommendations for users
def recommend_items(autoencoder, data, user_ids, item_ids, top_n=5):
    # Reconstruct the user-item matrix
    reconstructed = autoencoder.predict(data)

    recommendations = {}

    for user_idx, user_id in enumerate(user_ids):
        user_reconstructed_scores = reconstructed[user_idx]

        # Rank items by their predicted scores
        item_scores = list(zip(item_ids, user_reconstructed_scores))
        sorted_items = sorted(item_scores, key=lambda x: -x[1])

        # Exclude items the user has already interacted with (where playcount > 0)
        user_original_interactions = data[user_idx]
        recommended_items = [
            item
            for item, score in sorted_items
            if user_original_interactions[item_ids.tolist().index(item)] == 0
        ]

        recommendations[user_id] = recommended_items[:top_n]

    return recommendations


# Main function to execute the pipeline
def main():
    data_path = "../remappings/data/Modified_Listening_History.txt"

    # Load ratings data
    ratings = load_ratings_data(data_path)

    # Create user-item matrix (ratings matrix)
    user_item_matrix = create_user_item_matrix(ratings)

    # Normalize the data
    data_normalized, scaler = normalize_data(user_item_matrix.to_numpy())

    # Split the data into training and test sets
    train_data, test_data = train_test_split(
        data_normalized, test_size=0.2, random_state=42
    )

    # Train autoencoder with a smaller latent dimension (e.g., 5)
    latent_dim = 5  # Smaller latent dimension
    autoencoder = train_autoencoder(train_data, latent_dim)

    # Get recommendations for users in the test set
    user_ids = user_item_matrix.index
    item_ids = user_item_matrix.columns
    recommendations = recommend_items_with_similarity(
        autoencoder, data_normalized, user_ids, item_ids, top_n=5
    )

    # Print some example recommendations
    for user_id, recommended_items in recommendations.items():
        print(f"Recommended items for user {user_id}: {recommended_items}")


if __name__ == "__main__":
    main()
