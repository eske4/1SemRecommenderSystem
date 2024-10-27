import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sps
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

def read_data(file_path):
    data = pd.read_csv(
        file_path,
        sep='\t',
        dtype={'track_id': int, 'user_id': int, 'playcount': int},
        nrows=1000  # Adjust or remove nrows to read more data
    )

    # Inspect the first few rows to ensure correct reading
    print("Sampled Data:")
    print(data.head())

    # Binarize the play counts
    data['playcount'] = 1

    # Map IDs to zero-based indices
    unique_user_ids = data['user_id'].unique()
    unique_track_ids = data['track_id'].unique()

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    track_id_to_index = {track_id: idx for idx, track_id in enumerate(unique_track_ids)}

    # Apply the mappings to create index columns
    data['user_idx'] = data['user_id'].map(user_id_to_index)
    data['track_idx'] = data['track_id'].map(track_id_to_index)

    num_users = len(unique_user_ids)
    num_tracks = len(unique_track_ids)

    print(f"Dataset Size: {len(data)}")
    print(f"Number of users: {num_users}")
    print(f"Number of tracks: {num_tracks}")

    return data, num_users, num_tracks

def split_data(data):
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )

    return train_data, test_data

def create_user_song_matrix(train_data, num_users, num_tracks):
    #Create a sparse matrix with play counts
    V = sps.csr_matrix(
        (train_data['playcount'], (train_data['user_idx'], train_data['track_idx'])),
        shape=(num_users, num_tracks)
    )

    #Calculate and print matrix statistics
    total_possible = num_users * num_tracks
    non_zero_entries = V.nnz
    sparsity = 1 - (non_zero_entries / total_possible)

    print('Target:\n%s' % V.todense())
    print(f"Matrix Shape: {V.shape}")
    print(f"Non-zero Entries: {non_zero_entries}")
    print(f"Zero Entries: {total_possible - non_zero_entries}")
    print(f"Sparsity: {sparsity:.6f} ({sparsity*100:.2f}%)")
    return V

# Provide insights into the distribution and variability of play counts,
def compute_split_statistics(train_data, test_data):
    print(f"Training data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")
    train_values = train_data['playcount'].values
    mean_train = np.mean(train_values)
    median_train = np.median(train_values)
    std_train = np.std(train_values, ddof=1)
    print(f"Training values - Mean: {mean_train}, Median: {median_train}, Std Dev: {std_train}")

def factorize(V):
    nmf = NMF(
        n_components=20,
        init='nndsvda',
        solver='mu',  # Use 'mu' solver for sparse input
        beta_loss='frobenius',
        max_iter=500,
        random_state=42,
        # verbose=True
    )

    # Fit the NMF model to the sparse training data
    W = nmf.fit_transform(V)
    H = nmf.components_

    print('Basis matrix:\n%s' % W)
    print('Mixture matrix:\n%s' % H)
    # Target estimate
    V_estimated = np.dot(W, H)
    print('Target estimate (W * H):\n', V_estimated)

    return W, H

def evaluate(W, H, test_data):

    predictions = []
    actuals = []
    
    for row in test_data.itertuples():
        user_idx = row.user_idx
        track_idx = row.track_idx
        actual_playcount = row.playcount
        predicted_playcount = np.dot(W[user_idx, :], H[:, track_idx])
        predicted_playcount = max(predicted_playcount, 0)  # Ensure non-negative
        
        predictions.append(predicted_playcount)
        actuals.append(actual_playcount)
    
    rmse = root_mean_squared_error(actuals, predictions)
    print(f"RMSE: {rmse:.3f}")

def main():
    # Load data
    data, num_users, num_tracks = read_data('Modified_Listening_History.txt')

    # Split data
    train_data, test_data = split_data(data)
    compute_split_statistics(train_data, test_data)

    # Create user-song matrix
    V = create_user_song_matrix(train_data, num_users, num_tracks)
    
    # Factorize the matrix
    W, H = factorize(V)
    
    # Evaluate the model
    evaluate(W, H, test_data)

if __name__ == "__main__":
    main()