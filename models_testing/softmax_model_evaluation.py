import pandas as pd
from lenskit import topn

# Example DataFrames
# Ground truth interactions (test set)
truth = pd.DataFrame({
    'item': [101,102,3,4,5,6,7,8,9,10],
    'user': [1,1,1,1,1,1,1,1,1,1],
})

# Predicted recommendations from your system (both relevant items but ranked incorrectly)
predicted = pd.DataFrame({
    'item': [102, 101,30,40,50,60,70,80,90,100], 
    'user': [1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0],
    'rank': [1, 2,3,4,5,6,7,8,9,10],
})

print("truth: ", truth, "predicted: ", predicted)

# Initialize a RecListAnalysis object
rla = topn.RecListAnalysis()

# Add the precision@k, recall@k, and ndcg@k metrics to evaluate
rla.add_metric(topn.precision)
rla.add_metric(topn.recall)
rla.add_metric(topn.ndcg)

# Evaluate precision@k, recall@k, and ndcg@k (e.g., k=5)
results = rla.compute(predicted, truth)

# Extract the metrics for evaluation
precision_at_k = results['precision'].mean()
recall_at_k = results['recall'].mean()
ndcg_at_k = results['ndcg'].mean()

# Print out the evaluation results
print(f'Precision@5: {precision_at_k:.4f}')
print(f'Recall@5: {recall_at_k:.4f}')
print(f'NDCG@5: {ndcg_at_k:.4f}')
