import ast
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# Define Autoencoder model class
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
                layers.Dense(np.prod(shape), activation="sigmoid"),
                layers.Reshape(shape),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Load track features from file
def load_track_features(path, max_rows=None):
    return pd.read_csv(path, delimiter="\t", nrows=max_rows)


# Prepare data by normalizing it
def prepare_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler


# Preprocess features using onehot encoding method
def preprocess_features(data):
    # Separate and prepare the metadata
    metadata = data[["name", "artist", "track_id"]]

    # Drop metadata from feature data and handle categorical columns
    features = data.drop(columns=["name", "artist", "track_id"], errors="ignore")
    features["genre"] = features["genre"].astype("category")

    def parse_tags(x):
        if isinstance(x, str):
            # Check if the string looks like a list
            if x.startswith("[") and x.endswith("]"):
                # Convert from string representation of a list to a list
                return eval(
                    x
                )  # This could still be risky, consider using ast.literal_eval
            else:
                # Split by commas and strip whitespace
                return [tag.strip() for tag in x.split(",") if tag.strip()]
        return []

    features["tags"] = features["tags"].apply(parse_tags)

    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(features["tags"])

    genre_encoded = pd.get_dummies(features["genre"], prefix="genre")
    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_).add_prefix("tag_")
    # Rename columns based on mapping

    print(tags_df.head())
    print(genre_encoded.head())

    # Combine processed features with one-hot encoded data
    processed_data = pd.concat([features, genre_encoded, tags_df], axis=1)
    print(len(processed_data.head()))
    return processed_data.drop(columns=["genre", "tags", "year"]), metadata


# Train the autoencoder on training data
def train_autoencoder(train_data, latent_dim):
    input_shape = train_data.shape[1:]
    autoencoder = Autoencoder(latent_dim, input_shape)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(train_data, train_data, epochs=30, batch_size=512)
    return autoencoder


# Generate encoded representations of data
def encode_data(autoencoder, data):
    return autoencoder.encoder.predict(data)


# Recommend similar tracks based on cosine similarity
def get_recommendations(input_feature, encoded_data, items):
    # Calculate similarity scores
    sim_scores = cosine_similarity([input_feature], encoded_data)[0]

    # Sort the scores in descending order
    sorted_indices = np.argsort(sim_scores)[::-1]

    sorted_scores = sim_scores[sorted_indices]
    sorted_items = items.iloc[sorted_indices]

    return sorted_indices, sorted_scores, sorted_items


def filter_recommendations(
    filtered_indices, filtered_scores, sorted_items, min_score=0.5, max_score=0.7
):
    # Apply the range filter to include only scores between min_score and max_score
    range_mask = (filtered_scores > min_score) & (filtered_scores < max_score)
    filtered_indices = filtered_indices[range_mask]
    filtered_scores = filtered_scores[range_mask]
    sorted_items = sorted_items.iloc[np.where(range_mask)[0]]

    # Re-sort by filtered similarity scores in descending order
    final_sorted_order = np.argsort(filtered_scores)[::-1]
    filtered_indices = filtered_indices[final_sorted_order]
    filtered_scores = filtered_scores[final_sorted_order]
    sorted_items = sorted_items.iloc[final_sorted_order]

    return filtered_indices, filtered_scores, sorted_items


def randomize_items(filtered_indices, filtered_scores, sorted_items):

    # Randomize the order of the filtered recommendations
    randomized_order = np.random.permutation(len(filtered_indices))
    filtered_indices = filtered_indices[randomized_order]
    filtered_scores = filtered_scores[randomized_order]
    sorted_items = sorted_items.iloc[randomized_order]

    return filtered_indices, filtered_scores, sorted_items


# Display recommendations
def display_recommendations(track_id, encoded_data, test_data, items_with_metadata):
    similar_indices, similar_scores, item_with_metadata = get_recommendations(
        track_id, encoded_data, items_with_metadata
    )

    filtered_indices, filtered_scores, filtered_meta_items = filter_recommendations(
        similar_indices, similar_scores, items_with_metadata
    )

    filtered_indices, filtered_scores, filtered_meta_items = randomize_items(
        filtered_indices, filtered_scores, filtered_meta_items
    )

    print("Recommended Track Indices:", filtered_indices)
    print("Similarity Scores:", filtered_scores)
    print("recommended items:", filtered_meta_items)

    diverse_indices, diverse_scores, diverse_meta_items, max_similarities = (
        get_diverse_recommendations(
            encoded_data, track_id, items_with_metadata, top_n=10
        )
    )

    print("Top Recommended Indices:", diverse_indices)
    print("Similarity Scores:", diverse_scores)
    print("Recommended Items:", diverse_meta_items)
    print("Max Similarity for Each Item in List:", max_similarities)


# Refined function to get diverse recommendations based on feature vector
def get_diverse_recommendations(
    encoded_data, input_features, items, top_n=10, min_score=0.7, max_score=0.5
):
    # Calculate similarity scores between input_features and all encoded items
    sim_scores = cosine_similarity([input_features], encoded_data)[
        0
    ]  # Use input_features directly

    # TODO implement min max range for recommendations
    range_mask = (sim_scores > min_score) & (sim_scores < max_score)

    # Sort by similarity scores in descending order
    sorted_indices = np.argsort(sim_scores)[::-1]
    sorted_scores = sim_scores[sorted_indices]

    # Initialize top recommendations with the most similar item
    top_indices = [sorted_indices[0]]
    top_scores = [sorted_scores[0]]
    top_meta_items = [items.iloc[sorted_indices[0]]]
    max_sim = sorted_scores[0]  # Start max_sim with the first similarity score

    for _ in range(1, top_n):
        # Calculate maximum similarity to any previously selected item
        max_similarity_to_selected = np.zeros(len(sim_scores))
        for idx in top_indices:
            pairwise_similarities = cosine_similarity(
                [encoded_data[idx]], encoded_data
            ).flatten()
            max_similarity_to_selected = np.maximum(
                max_similarity_to_selected, pairwise_similarities
            )

        # Exclude already selected items by setting their similarity to infinity
        max_similarity_to_selected[top_indices] = np.inf

        # Select the item with the lowest maximum similarity to the selected set
        next_index = np.argmin(max_similarity_to_selected)

        # Append the selected item and its metadata
        top_indices.append(next_index)
        top_scores.append(sim_scores[next_index])
        top_meta_items.append(items.iloc[next_index])

        # Update max_sim to be the highest similarity among all selected items
        max_sim = max(max_sim, max_similarity_to_selected[next_index])

    # Convert lists to NumPy arrays and keep metadata as a DataFrame
    indices = np.array(top_indices)
    scores = np.array(top_scores)
    meta_items_df = pd.DataFrame(top_meta_items)

    return indices, scores, meta_items_df, max_sim


# Main function to execute the pipeline
def main():
    data_path = "../remappings/data/Modified_Music_info.txt"
    raw_data = load_track_features(data_path)

    # Preprocess data
    final_data, meta_data = preprocess_features(raw_data)

    # Normalize and split data
    data_normalized, scaler = prepare_data(final_data.to_numpy())
    train_data, test_data = train_test_split(
        data_normalized, test_size=0.2, random_state=42
    )

    # Train autoencoder
    latent_dim = 90
    autoencoder = train_autoencoder(train_data, latent_dim)

    encoded_test = encode_data(autoencoder, data_normalized)
    decoded_test = autoencoder.decoder.predict(encoded_test)

    # Convert decoded output to DataFrame
    decoded_test_df = pd.DataFrame(decoded_test, columns=final_data.columns)

    # Combine decoded data with metadata
    combined_data = pd.concat(
        [meta_data.reset_index(drop=True), decoded_test_df], axis=1
    )

    # Display recommendations
    input_feature = decoded_test[0]
    display_recommendations(input_feature, decoded_test, data_normalized, combined_data)


if __name__ == "__main__":
    main()
