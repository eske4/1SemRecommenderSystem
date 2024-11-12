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
    metadata = data[["user_id", "track_id"]]
    ratings = data[["playcount"]]

    return ratings, metadata


# Train the autoencoder on training data
def train_autoencoder(train_data, latent_dim):
    input_shape = train_data.shape[1:]
    autoencoder = Autoencoder(latent_dim, input_shape)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(train_data, train_data, epochs=5, batch_size=32)
    return autoencoder


# Generate encoded representations of data
def encode_data(autoencoder, data):
    return autoencoder.encoder.predict(data)


# Main function to execute the pipeline
def main():
    data_path = "../remappings/data/Modified_Listening_History.txt"
    raw_data = load_track_features(data_path)

    # Preprocess data
    final_data, meta_data = preprocess_features(raw_data)

    # Normalize and split data
    data_normalized, scaler = prepare_data(final_data.to_numpy())
    train_data, test_data = train_test_split(
        data_normalized, test_size=0.2, random_state=42
    )

    # Train autoencoder
    latent_dim = 30
    autoencoder = train_autoencoder(train_data, latent_dim)

    encoded_test = encode_data(autoencoder, data_normalized)
    decoded_test = autoencoder.decoder.predict(encoded_test)

    # Convert decoded output to DataFrame
    decoded_test_df = pd.DataFrame(decoded_test, columns=final_data.columns)

    # Combine decoded data with metadata
    combined_data = pd.concat(
        [meta_data.reset_index(drop=True), decoded_test_df], axis=1
    )


if __name__ == "__main__":
    main()
