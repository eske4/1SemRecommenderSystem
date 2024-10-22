import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

def load_interactions(file_path, chunksize=10**6):
    #These lists will store the numerical indices for users and songs, and the corresponding play counts
    user_indices = []
    song_indices = []
    play_counts = []
    # These dictionaries will map the original user_id and song_id to unique numerical indices
    user_id_to_index = {}
    song_id_to_index = {}
    # Counters to assign unique numerical indices to users and songs
    user_index = 0
    song_index = 0

    # Reads the CSV File in Chunks
    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype={'user_id': str, 'track_id': str, 'playcount': int}, low_memory=False):
        for _, row in chunk.iterrows():
            user_id = row['user_id']
            song_id = row['track_id']
            play_count = row['playcount']

            # Map user_id to user_index
            if user_id not in user_id_to_index:
                user_id_to_index[user_id] = user_index
                user_index += 1
            u_idx = user_id_to_index[user_id]

            # Map song_id to song_index
            if song_id not in song_id_to_index:
                song_id_to_index[song_id] = song_index
                song_index += 1
            s_idx = song_id_to_index[song_id]

            # Collect data
            user_indices.append(u_idx)
            song_indices.append(s_idx)
            play_counts.append(play_count)

    return user_indices, song_indices, play_counts, user_id_to_index, song_id_to_index

# Calculates and displays statistics about the sparsity of the user-song interaction matrix
def compute_sparsity_statistics(interactions_df, num_users, num_songs):
    total_entries = num_users * num_songs
    num_nonzero_original = len(interactions_df)
    num_zero_original = total_entries - num_nonzero_original
    density = num_nonzero_original / total_entries
    sparsity = 1 - density
    
    print(f"Total entries in matrix: {total_entries}")
    print(f"Non-zero entries: {num_nonzero_original}")
    print(f"Zero entries: {num_zero_original}")
    print(f"Density: {density:.4%}")
    print(f"Sparsity: {sparsity:.4%}")

# Provide insights into the distribution and variability of play counts,
def compute_training_statistics(train_interactions):
    train_values = train_interactions['play_count'].values
    mean_train = np.mean(train_values)
    median_train = np.median(train_values)
    std_train = np.std(train_values, ddof=1)
    print(f"Training values - Mean: {mean_train}, Median: {median_train}, Std Dev: {std_train}")

# Main code
# Load interactions
user_indices, song_indices, play_counts, user_id_to_index, song_id_to_index = load_interactions('User Listening History 100K.csv')

# Create a DataFrame of interactions
interactions_df = pd.DataFrame({
    'user_idx': user_indices,
    'song_idx': song_indices,
    'play_count': play_counts
})

# Binarize the play counts
interactions_df['play_count'] = 1

# Compute sparsity statistics
num_users = len(user_id_to_index)
num_songs = len(song_id_to_index)
compute_sparsity_statistics(interactions_df, num_users, num_songs)

# Split interactions into training and test sets
train_interactions, test_interactions = train_test_split(
    interactions_df,
    test_size=0.2,
    random_state=42
)

# Compute training statistics
compute_training_statistics(train_interactions)

# Create sparse matrices for training and testing
V_train_sparse = sps.csr_matrix(
    (train_interactions['play_count'], (train_interactions['user_idx'], train_interactions['song_idx'])),
    shape=(num_users, num_songs)
)

# Initialize variables
best_rmse = float('inf')
best_rank = None
best_W, best_H = None, None

# Iterate over a range of ranks
for rank in range(5, 20):
    nmf_model = NMF(
        n_components=rank,
        init='nndsvda',
        solver='cd',  # Use 'mu' solver for sparse input
        beta_loss='frobenius',
        max_iter=500,
        random_state=0,
        # verbose=True
    )

    # Fit the NMF model to the sparse training data
    W = nmf_model.fit_transform(V_train_sparse)
    H = nmf_model.components_
    
    # Predict test values
    test_user_indices = test_interactions['user_idx'].values
    test_song_indices = test_interactions['song_idx'].values
    test_play_counts = test_interactions['play_count'].values
    
    predicted_values = np.sum(W[test_user_indices, :] * H[:, test_song_indices].T, axis=1)
    
    # Clip predicted values to the range [0, 1] since we're dealing with binary data
    predicted_values = np.clip(predicted_values, 0, 1)

    # Inspect Predicted Values for Test Set
    print(f"First 10 predicted values: {predicted_values[:10]}")
    print(f"First 10 actual test values: {test_play_counts[:10]}")
    
    # Check Variability in W and H Matrices
    W_norm = np.linalg.norm(W)
    H_norm = np.linalg.norm(H)
    print(f"Rank: {rank}, W norm: {W_norm}, H norm: {H_norm}")
    
    # Compute RMSE on test set
    rmse = root_mean_squared_error(test_play_counts, predicted_values)
    print(f"Rank: {rank}, RMSE on test set: {rmse}")
    
    # Update best model
    if rmse < best_rmse:
        best_rmse = rmse
        best_rank = rank
        best_W, best_H = W, H

print(f"Best Rank: {best_rank}, Best RMSE: {best_rmse}")
