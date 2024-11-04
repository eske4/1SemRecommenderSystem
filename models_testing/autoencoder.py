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
    return processed_data.drop(columns=["genre", "tags"]), metadata


# Train the autoencoder on training data
def train_autoencoder(train_data, latent_dim):
    input_shape = train_data.shape[1:]
    autoencoder = Autoencoder(latent_dim, input_shape)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(train_data, train_data, epochs=10)
    return autoencoder


# Generate encoded representations of data
def encode_data(autoencoder, data):
    return autoencoder.encoder.predict(data)


# Recommend similar tracks based on cosine similarity
def recommend_similar_tracks(track_id, encoded_data, items):
    sim_scores = cosine_similarity([encoded_data[track_id]], encoded_data)[0]
    sorted_indices = np.argsort(sim_scores)[::-1]

    # Exclude the selected track itself from recommendations
    mask = sorted_indices != track_id
    filtered_indices = sorted_indices[mask]
    filtered_scores = sim_scores[mask]
    sorted_items = items.iloc[filtered_indices]

    return filtered_indices, filtered_scores, sorted_items


# Display recommendations
def display_recommendations(track_id, encoded_data, test_data, items_with_metadata):
    similar_indices, similar_scores, item_with_metadata = recommend_similar_tracks(
        track_id, encoded_data, items_with_metadata
    )

    print("Recommended Track Indices:", similar_indices)
    print("Similarity Scores:", similar_scores)
    print("recommended items:", item_with_metadata)


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
    latent_dim = 2
    autoencoder = train_autoencoder(train_data, latent_dim)

    encoded_test = encode_data(autoencoder, data_normalized)
    decoded_test = autoencoder.decoder.predict(encoded_test)

    # Convert decoded output to DataFrame
    decoded_test_df = pd.DataFrame(decoded_test, columns=final_data.columns)

    # Combine decoded data with metadata
    combined_data = pd.concat(
        [meta_data.reset_index(drop=True), decoded_test_df], axis=1
    )

    print(combined_data)

    # Display recommendations
    track_id_to_recommend = 0
    display_recommendations(
        track_id_to_recommend, encoded_test, data_normalized, combined_data
    )


if __name__ == "__main__":
    main()
