import numpy as np
from itertools import combinations
from scipy.spatial.distance import cosine

def calculate_recommendation_diversity(recommended_items, V):
    """
    Calculate the diversity of a list of recommended items.
    
    Diversity is measured as the average pairwise cosine distance between the item vectors in V.
    
    Parameters:
        - recommended_items: List of recommended item indices
        - V: Item matrix (latent feature vectors for items)
    
    Returns:
        - diversity_score: Average pairwise cosine distance between recommended items
    """
    if len(recommended_items) < 2:
        return 0  # No diversity possible with fewer than 2 items
    
    # Get the latent vectors for the recommended items
    item_vectors = [V[item_id] for item_id, _ in recommended_items]
    
    # Calculate pairwise cosine distances
    pairwise_distances = []
    for vec1, vec2 in combinations(item_vectors, 2):
        distance = cosine(vec1, vec2)  # Cosine distance between two item vectors
        pairwise_distances.append(distance)
    
    # Calculate average diversity (average pairwise distance)
    diversity_score = np.mean(pairwise_distances)
    
    return diversity_score

def matrix_factorization(R, K, steps, learning_rate, reg_param):
    """
    Perform matrix factorization using stochastic gradient descent.

    Parameters:
       - R: User-Item matrix
       - K: Number of latent features
       - steps: Number of iterations
       - learning_rate: Learning rate for SGD
       - reg_param: Regularization parameter

    Returns:
       - U: User matrix
       - V: Item matrix
       - RMSE: Root Mean Squared Error
    """

    # Random initialization of user and item matrices
    num_users, num_items = R.shape
    
    # Random initialization of user and item matrices
    U = np.random.rand(num_users, K)
    V = np.random.rand(num_items, K)

    # Stochastic Gradient Descent
    for step in range(steps):
        for i in range(num_users):
            for j in range(num_items):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(U[i, :], V[j, :].T)
                    for k in range(K):
                        U[i][k] += learning_rate * (2 * eij * V[j][k] - reg_param * U[i][k])
                        V[j][k] += learning_rate * (2 * eij * U[i][k] - reg_param * V[j][k])

        # Calculate RMSE for monitoring
        error = 0
        for i in range(num_users):
            for j in range(num_items):
                if R[i][j] > 0:
                    error += (R[i][j] - np.dot(U[i, :], V[j, :].T))**2
                    for k in range(K):
                        error += (reg_param / 2) * (U[i][k]**2 + V[j][k]**2)

        rmse = np.sqrt(error / len(R[R > 0]))

        # Print RMSE at intervals
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{steps}, RMSE: {rmse}")

    return U, V, rmse

def matrix_factorization_with_diversity(R, K, steps, learning_rate, reg_param, diversity_weight):
    """
    Perform matrix factorization using SGD with an additional diversity function.
    
    Parameters:
       - R: User-Item matrix
       - K: Number of latent features
       - steps: Number of iterations
       - learning_rate: Learning rate for SGD
       - reg_param: Regularization parameter for U and V matrices
       - diversity_weight: Weight for the diversity penalty term (regularization)

    Returns:
       - U: User matrix
       - V: Item matrix
       - RMSE: Root Mean Squared Error
    """

    num_users, num_items = R.shape
    
    # Random initialization of user and item matrices
    U = np.random.rand(num_users, K)
    V = np.random.rand(num_items, K)

    # Stochastic Gradient Descent
    for step in range(steps):
        for i in range(num_users):
            for j in range(num_items):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(U[i, :], V[j, :].T)
                    for k in range(K):
                        U[i][k] += learning_rate * (2 * eij * V[j][k] - reg_param * U[i][k])
                        V[j][k] += learning_rate * (2 * eij * U[i][k] - reg_param * V[j][k])

        # Diversity regularization: Reduce similarity between item vectors
        if (diversity_weight != 0):
            for j1 in range(num_items):
                for j2 in range(j1 + 1, num_items):
                    similarity = np.dot(V[j1, :], V[j2, :]) / (np.linalg.norm(V[j1, :]) * np.linalg.norm(V[j2, :]) + 1e-10)  # Add small value to avoid divide by zero
                    
                    # Smoother diversity penalty, rescaling similarity to a smaller range
                    diversity_penalty = diversity_weight * (1 - similarity) / K

                    for k in range(K):
                        update_value = learning_rate * diversity_penalty * (V[j1][k] - V[j2][k])

                        # Clip the update value to prevent overflow
                        update_value = np.clip(update_value, -0.01, 0.01)

                        V[j1][k] += update_value
                        V[j2][k] -= update_value

        # Calculate RMSE for monitoring
        error = 0
        for i in range(num_users):
            for j in range(num_items):
                if R[i][j] > 0:
                    error += (R[i][j] - np.dot(U[i, :], V[j, :].T))**2
                    for k in range(K):
                        error += (reg_param / 2) * (U[i][k]**2 + V[j][k]**2)

        rmse = np.sqrt(error / len(R[R > 0]))

        # Print RMSE at intervals
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{steps}, RMSE: {rmse}")

    return U, V, rmse

# After training the model and obtaining matrices U and V

def recommend_item(user_id, U, V, R, min_rating=1, max_rating=5):
    """
    Recommend items for a specific user.

    Parameters:
       - user_id: User for whom recommendations are needed
       - U: User matrix
       - V: Item matrix
       - R: User-item matrix for checking already rated items
       - min_rating: Minimum rating in the scale
       - max_rating: Maximum rating in the scale

    Returns:
       - recommendations: List of recommended items
    """
    user_vector = U[user_id, :]  # Get the user vector from the user matrix
    predicted_ratings = np.dot(user_vector, V.T)  # Predict ratings for all items

    # Clamp the predicted ratings to be within the valid rating range
    predicted_ratings = np.clip(predicted_ratings, min_rating, max_rating)

    # Find the indices of unrated items for the user
    unrated_indices = np.where(R[user_id, :] == 0)[0]

    # Sort the unrated items based on predicted ratings
    recommended_items = sorted(zip(unrated_indices, predicted_ratings[unrated_indices]), key=lambda x: x[1], reverse=True)

    return recommended_items

# A larger user-item matrix (10 users, 10 items)
R = np.array([
    [5, 3, 0, 1, 4, 0, 0, 3, 0, 0],
    [4, 0, 0, 1, 0, 0, 5, 0, 3, 2],
    [1, 1, 0, 5, 0, 0, 4, 0, 0, 4],
    [0, 0, 0, 4, 0, 5, 0, 0, 4, 0],
    [0, 1, 5, 4, 0, 3, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 5, 4, 4, 0, 1],
    [1, 0, 2, 0, 4, 0, 0, 5, 0, 0],
    [0, 5, 0, 4, 0, 0, 4, 0, 5, 1],
    [3, 0, 4, 0, 5, 0, 0, 0, 4, 0],
    [5, 3, 0, 1, 0, 0, 4, 0, 3, 0]
])


# Set parameters
K = 3  # Number of latent features
steps = 5000
learning_rate = 0.01
reg_param = 0.04
diversity_weight = 0.04  # Strength of diversity regularization

# Perform matrix factorization with diversity
U, V, rmse = matrix_factorization_with_diversity(R, K, steps, learning_rate, reg_param, diversity_weight)

# Recommend and evaluate diversity
user_id_to_recommend = 0  # The user we want to recommend for
recommended_items = recommend_item(user_id_to_recommend, U, V, R, min_rating=1, max_rating=5)

# Calculate diversity for the top N recommendations
top_n = 2
top_recommendations = recommended_items[:top_n]
diversity_score = calculate_recommendation_diversity(top_recommendations, V)

# Display top N recommendations and diversity score
print(f"\nTop {top_n} recommendations for User {user_id_to_recommend}:")
for item_id, predicted_rating in top_recommendations:
    print(f"Item: {item_id}, Predicted Rating: {int(predicted_rating)}")

print(f"\nDiversity score of the recommendations: {diversity_score:.4f}")