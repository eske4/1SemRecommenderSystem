import numpy as np
import pandas as pd
from autoencoder import Autoencoder
from ranking_metrics import RankingMetrics
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer


# Load track features from file
def load_track_features(path, max_rows=None):
    return pd.read_csv(path, delimiter="\t", nrows=max_rows)


# Prepare data by normalizing it
def prepare_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler


def preprocess_features(data):
    metadata = data[["name", "artist", "track_id"]]
    features = data.drop(columns=["name", "artist", "track_id"], errors="ignore")
    features["genre"] = features["genre"].astype("category")

    features["tags"] = features["tags"].apply(
        lambda x: (
            eval(x)
            if isinstance(x, str) and x.startswith("[") and x.endswith("]")
            else (
                [tag.strip() for tag in x.split(",") if tag.strip()]
                if isinstance(x, str)
                else []
            )
        )
    )

    mlb = MultiLabelBinarizer()
    tags_encoded = pd.DataFrame(
        mlb.fit_transform(features["tags"]), columns=mlb.classes_
    ).add_prefix("tag_")
    genre_encoded = pd.get_dummies(features["genre"], prefix="genre")

    processed_data = pd.concat([features, genre_encoded, tags_encoded], axis=1)
    return processed_data.drop(columns=["genre", "tags", "year"]), metadata


def aggregate_user_preference(user_id, ratings, tracks, autoencoder):
    """
    Aggregates user preferences by filtering the tracks based on user ratings and passing them through the autoencoder.

    Parameters:
        user_id (int): The user ID to filter the tracks.
        ratings (pd.DataFrame): The ratings dataset containing user-item interactions.
        tracks (pd.DataFrame): The tracks dataset containing track features.
        autoencoder (tf.keras.Model): The trained autoencoder model.

    Returns:
        np.ndarray: Aggregated user preferences after decoding the autoencoder's output.
    """
    # Step 1: Get the track IDs that the user has rated
    user_ratings = ratings[ratings["user_id"] == user_id]
    user_track_ids = user_ratings["track_id"].values

    # Step 3: Use iloc to select the rows from the tracks dataframe based on the indices
    user_tracks = tracks.iloc[user_track_ids]

    # Step 8: Aggregate the decoded data (e.g., by averaging across all tracks)
    aggregated_preference = np.mean(user_tracks, axis=0)

    return aggregated_preference


def get_all_user(ratings):
    return ratings["user_id"].unique()


def get_rated_list(user_id, ratings):
    user_ratings = ratings[ratings["user_id"] == user_id]
    return user_ratings["track_id"].values


def average_intra_list_distance(items, distance_metric="cosine"):
    """
    Calculate the Average Intra-List Distance (AILD).

    Parameters:
        items (pd.DataFrame or np.ndarray): A DataFrame or array of items (vectors).
        distance_metric (str): The distance metric to use ('euclidean' or 'cosine').

    Returns:
        float: The AILD score.
    """
    # Convert items to NumPy array if it's a DataFrame
    if isinstance(items, pd.DataFrame):
        items = items.to_numpy()

    n = len(items)
    if n < 2:
        return 0  # AILD is undefined for lists with fewer than 2 items

    # Select the appropriate distance function
    if distance_metric == "euclidean":
        dist_func = euclidean
    elif distance_metric == "cosine":
        dist_func = cosine
    else:
        raise ValueError("Unsupported distance metric. Choose 'euclidean' or 'cosine'.")

    # Compute all pairwise distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(
                dist_func(items[i], items[j])
            )  # Correctly access rows as vectors

    # Calculate the average distance
    return np.mean(distances)


# Train the autoencoder on training data
def train_autoencoder(train_data, latent_dim):
    input_shape = train_data.shape[1:]
    autoencoder = Autoencoder(latent_dim, input_shape)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(train_data, train_data, epochs=30, batch_size=512)
    return autoencoder


# Generate encoded representations of data
def encode_data(autoencoder, data):
    encoded_data = autoencoder.encoder.predict(data)
    decoded_data = autoencoder.decoder.predict(encoded_data)
    return encoded_data, decoded_data


# Recommend similar tracks based on cosine similarity
def sort_by_distance(input_feature, encoded_data, items):
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
    similar_indices, similar_scores, item_with_metadata = sort_by_distance(
        track_id, encoded_data, items_with_metadata
    )

    filtered_indices, filtered_scores, filtered_meta_items = filter_recommendations(
        similar_indices, similar_scores, items_with_metadata
    )

    diverse_indices, diverse_scores, diverse_meta_items, max_similarities = (
        get_diverse_recommendations(
            encoded_data, track_id, items_with_metadata, top_n=10
        )
    )

    numeric_data = diverse_meta_items.drop(
        columns=["name", "artist", "track_id"], errors="ignore"
    )
    return similar_indices, diverse_indices


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
    tracks_path = "../../remappings/data/Modified_Music_info.txt"
    raw_tracks = load_track_features(tracks_path)

    ratings_path = "../../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt"
    raw_ratings = load_track_features(ratings_path)

    # Preprocess data
    final_data, meta_data = preprocess_features(raw_tracks)

    # Normalize and split data
    data_normalized, scaler = prepare_data(final_data.to_numpy())
    train_data, test_data = train_test_split(
        data_normalized, test_size=0.2, random_state=42
    )

    # Train autoencoder
    latent_dim = 90
    autoencoder = train_autoencoder(train_data, latent_dim)

    encoded_test, decoded_test = encode_data(autoencoder, data_normalized)

    # Convert decoded output to DataFrame
    decoded_test_df = pd.DataFrame(decoded_test, columns=final_data.columns)

    # Combine decoded data with metadata
    combined_data = pd.concat(
        [meta_data.reset_index(drop=True), decoded_test_df], axis=1
    )

    user_ids = get_all_user(raw_ratings)

    for user in user_ids:

        input_feature = aggregate_user_preference(
            user_id=user,
            ratings=raw_ratings,
            tracks=decoded_test_df,
            autoencoder=autoencoder,
        )
        user_ratings = get_rated_list(user_id=user, ratings=raw_ratings)

        similar_indices, diverse_indices = display_recommendations(
            input_feature, decoded_test, data_normalized, combined_data
        )

        # Initialize the metrics calculator
        metrics = RankingMetrics(similar_indices, user_ratings)

        # Calculate metrics
        k = 10
        summary = metrics.metrics_summary(k)

        # Print results
        print(f"user: {user} Metrics Summary@{k}: {summary}")


if __name__ == "__main__":
    main()
