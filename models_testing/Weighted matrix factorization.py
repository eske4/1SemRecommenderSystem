import numpy as np

# Function to perform weighted matrix factorization
def weighted_matrix_factorization(R, W, num_factors, steps, learning_rate, regularization):
    # Get the shape of the user-item array so we know how many users and items there are
    num_users, num_items = R.shape
    # Initialize user and item latent factor matrices with random values
    U = np.random.rand(num_users, num_factors)
    V = np.random.rand(num_items, num_factors)

    # Perform optimization for the given number of steps
    for step in range(steps):
        # Iterate over each user-item pair
        for i in range(num_users):
            for j in range(num_items):
                if W[i, j] > 0:  # Only update for non-zero weights
                    error_ij = R[i, j] - np.dot(U[i, :], V[j, :].T)

                    # Update latent factors using gradient descent
                    # Learning rate and the regularization parameter are used to make sure U and V don't grow too large and therefore cause overfitting
                    for k in range(num_factors):
                        U[i, k] += learning_rate * (2 * W[i, j] * error_ij * V[j, k] - regularization * U[i, k])
                        V[j, k] += learning_rate * (2 * W[i, j] * error_ij * U[i, k] - regularization * V[j, k])

        # Calculate the total loss (error) at each step
        loss = 0
        for i in range(num_users):
            for j in range(num_items):
                if W[i, j] > 0:  # Only consider weighted ratings
                    loss += W[i, j] * (R[i, j] - np.dot(U[i, :], V[j, :].T)) ** 2
                    for k in range(num_factors):
                        loss += (regularization / 2) * (U[i, k] ** 2 + V[j, k] ** 2)
        
        # Print loss at every 1000th step to monitor progress
        if step % 1000 == 0:
            print(f"Step {step}/{steps}, loss: {loss:.4f}")

    return U, V

# RMSE calculation function
def rmse(R, predicted_R, W):
    # Only consider the observed ratings (where W == 1)
    observed_indices = np.where(W == 1)
    error = R[observed_indices] - predicted_R[observed_indices]
    return np.sqrt(np.mean(error ** 2))

# Sample data: user-item matrix (10 users, 5 items)
# 0 indicates missing ratings
R = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 0],
    [1, 1, 0, 5, 0],
    [0, 0, 0, 4, 0],
    [0, 1, 5, 4, 0],
    [0, 3, 0, 2, 0],
    [1, 0, 2, 0, 4],
    [0, 5, 0, 4, 0],
    [3, 0, 4, 0, 5],
    [5, 3, 0, 1, 0]
])

# Weight matrix (W), where 1 means the rating is observed, and 0.01 means it is missing but softly considered
W = np.where(R > 0, 1, 0.01)  # Assign 0.01 weight to missing ratings

# Parameters
num_factors = 3
steps = 5000
learning_rate = 0.01
regularization = 0.04

# Perform matrix factorization
U, V = weighted_matrix_factorization(R, W, num_factors=num_factors, steps=steps, learning_rate=learning_rate, regularization=regularization)

# Predicted user-item matrix (reconstructed)
predicted_R = np.dot(U, V.T)

print("\nOriginal Ratings Matrix (with missing values):")
print(R)

print("\nPredicted Ratings Matrix:")
print(np.round(predicted_R, 2))

# Calculate RMSE
rmse_value = rmse(R, predicted_R, W)
print(f"RMSE: {rmse_value}")
