import os
import pandas as pd
from lenskit import topn
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k, diversity

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class ContentEvaluation:
    def __init__(self):
        self.truth_data = pd.read_csv("../../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt", delimiter="\t",)
        self.truth_data.rename(columns={'track_id': 'item', 'user_id': 'user'}, inplace=True)
        self.train_data = pd.read_csv("../../remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt", delimiter="\t",)
        self.train_data.rename(columns={'track_id': 'item', 'user_id': 'user'}, inplace=True)

    def LenskitEvaluation(self, predictions: pd.DataFrame) -> tuple:
        '''
        predictions = pd.DataFrame({
            'item': [Int], 
            'user': [Int],
            'rank': [Int],
        })
        '''
        # Ensure consistent data types for 'user' and 'item'
        predictions['user'] = predictions['user'].astype(str)
        predictions['item'] = predictions['item'].astype(str)
        self.truth_data['user'] = self.truth_data['user'].astype(str)
        self.truth_data['item'] = self.truth_data['item'].astype(str)

        # Ensure 'rank' is numeric (either int or float)
        if not pd.api.types.is_numeric_dtype(predictions['rank']):
            predictions['rank'] = predictions['rank'].astype(float)

        # Futurewarning at ..\lenskit\metrics\topn.py line 100 and 149. replace lines with "scores['ngood'] = scores['ngood'].fillna(0)"

        rla = topn.RecListAnalysis()
        rla.add_metric(topn.precision)
        rla.add_metric(topn.recall)
        rla.add_metric(topn.hit)
        rla.add_metric(topn.ndcg)
        
        # Compute the results
        results = rla.compute(predictions, self.truth_data)
        
        # Calculate the mean of each metric
        precision_score = results['precision'].mean()
        recall_score = results['recall'].mean()
        hit_score = results['hit'].mean()
        ndcg_score = results['ndcg'].mean()

        return (precision_score, recall_score, hit_score, ndcg_score)

    
    def RecommenderEvaluation(self, predictions: pd.DataFrame, k_value: int) -> tuple:
        '''
        predictions = pd.DataFrame({
            'item': [Int], 
            'user': [Int],
            'rank': [Int],
        })
        '''
        # Ensure consistent data types for 'user' and 'item'
        predictions['user'] = predictions['user'].astype(str)
        predictions['item'] = predictions['item'].astype(str)
        self.truth_data['user'] = self.truth_data['user'].astype(str)
        self.truth_data['item'] = self.truth_data['item'].astype(str)
        self.train_data['user'] = self.train_data['user'].astype(str)
        self.train_data['item'] = self.train_data['item'].astype(str)

        # Rename 'rank' column to 'prediction'
        predictions = predictions.rename(columns={'rank': 'prediction'})

        # Compute metrics
        precision_score = precision_at_k(
            self.truth_data, predictions, 
            col_user="user", col_item="item", col_prediction="prediction", k=k_value
        )
        recall_score = recall_at_k(
            self.truth_data, predictions, 
            col_user="user", col_item="item", col_prediction="prediction", k=k_value
        )
        ndcg_score = ndcg_at_k(
            self.truth_data, predictions, 
            col_user="user", col_item="item", col_rating="playcount", col_prediction="prediction", k=k_value
        )
        map_score = map(
            self.truth_data, predictions, 
            col_user="user", col_item="item", col_prediction="prediction", k=k_value, relevancy_method=None
        )

        eval_diversity = diversity(train_df=self.train_data,
                           reco_df=predictions,
                           col_user="user",
                           col_item="item")

        return (precision_score, recall_score, ndcg_score, map_score, eval_diversity)


    
