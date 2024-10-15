import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import root_mean_squared_error
from tabulate import tabulate

# Read the User-Song Data
df = pd.read_csv('user_song_matrix_sample.csv', index_col=0)

# Compute the Frobenius Norm of the original matrix
V = df.values  # Convert the DataFrame to a NumPy array (V represents the original matrix)
frobenius_norm = np.linalg.norm(V, 'fro')
benchmark_value = frobenius_norm * 0.0001  # Set the benchmark value. Means "Stop iterating when the reconstruction error is less than this small fraction of the overall matrix's magnitude."

# Initialize variables for the iterative process
rank = 15  # Starting rank (latent features)
rmse = float('inf')  # Initialize RMSE with a high value
best_rank = None
best_W, best_H = None, None
best_reconstructed = None

# Iterate through ranks and fit NMF
while rmse > benchmark_value:
    # Step 1: Initialize the NMF model with the current rank
    nmf_model = NMF(n_components=rank, init='nndsvd', solver='cd', random_state=0, max_iter=200, tol=1e-4, verbose=True)

    # Step 2: Fit the NMF model to the user-song matrix
    W = nmf_model.fit_transform(V)  # User features matrix
    H = nmf_model.components_  # Song features matrix

    # Step 2: Reconstruct the matrix
    reconstructed_matrix = np.dot(W, H)  # V ≈ W * H

    # Step 3: Calculate the RMSE between the original matrix and the reconstructed matrix
    rmse = root_mean_squared_error(V, reconstructed_matrix)

    # Step 4: If RMSE is less than the benchmark value, store the results
    if rmse < benchmark_value:
        best_rank = rank
        best_W, best_H = W, H
        best_reconstructed = reconstructed_matrix

    # Print the progress
    print(f"Rank: {rank}, RMSE: {rmse}")

    # Increment the rank
    rank += 1

# Output the best rank and the final reconstructed matrix
print(f"Best Rank: {best_rank}")

# Print a simplified portion of the reconstructed matrix
def print_simple_matrix(matrix, rows=5, cols=5):
    matrix_portion = pd.DataFrame(matrix).iloc[:rows, :cols]
    print(tabulate(matrix_portion, tablefmt='plain', showindex=False, numalign='right'))

print("Reconstructed Matrix at Best Rank:")
print_simple_matrix(best_reconstructed, rows=15, cols=15)

# Save the reconstructed matrix to a CSV file
# reconstructed_df = pd.DataFrame(best_reconstructed, index=df.index, columns=df.columns)
# reconstructed_df.to_csv('reconstructed_matrix_best_rank.csv')
# print("Reconstructed matrix saved as 'reconstructed_matrix_best_rank.csv'.")

# Print the number of non-zero values (for verification)
non_zero_values_original = np.count_nonzero(V)
print(f"Non-zero values in the original matrix: {non_zero_values_original}")

# Count non-zero values in the reconstructed matrix (strictly non-zero)
non_zero_values_reconstructed = np.count_nonzero(best_reconstructed)
print(f"Non-zero values in the reconstructed matrix: {non_zero_values_reconstructed}")