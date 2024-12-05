import pandas as pd
from lenskit import topn
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k

class ContentEvaluation:
    def __init__(self):
        self.truth_data = pd.read_csv('remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt', delimiter='\t')
        self.truth_data.rename(columns={'track_id': 'item', 'user_id': 'user'}, inplace=True)
        self.test_truth_data = pd.DataFrame({
            'user': [1, 1, 2, 2, 3, 3],
            'item': [101, 102, 103, 104, 105, 106],
            'playcount': [1, 1, 1, 1, 1, 1]
        })

    def LenskitEvaluation(self, predictions: pd.DataFrame) -> tuple:
        '''
        predictions = pd.DataFrame({
            'item': [Int], 
            'user': [Int],
            'rank': [Int],
        })
        '''
        self.truth_data['user'] = self.truth_data['user'].astype(int)
        predictions['user'] = predictions['user'].astype(int)

        # Futurewarning at ..\lenskit\metrics\topn.py line 100 and 149. replace lines with "scores['ngood'] = scores['ngood'].fillna(0)"

        rla = topn.RecListAnalysis()
        rla.add_metric(topn.precision)
        rla.add_metric(topn.recall)
        rla.add_metric(topn.hit)
        rla.add_metric(topn.ndcg)
        
        results = rla.compute(predictions, self.truth_data)
        precision_score = results['precision'].mean()
        recall_score = results['recall'].mean()
        hit_score= results['hit'].mean()
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
        # Futurewarning at ..\recommenders\evaluation\python_evaluation.py line 438 to 440. replace lines with "df_hit.groupby(col_user, as_index=False)[col_user].agg([("hit", "count")]), rating_true_common.groupby(col_user, as_index=False)[col_user].agg([("hit", "count")]),"
        predictions = predictions.rename(columns={'rank': 'prediction'})



        precision_score = precision_at_k(self.truth_data, predictions, col_user="user", col_item="item", col_prediction="prediction", k=k_value)
        recall_score = recall_at_k(self.truth_data, predictions, col_user="user", col_item="item", col_prediction="prediction", k=k_value)
        ndcg_score = ndcg_at_k(self.truth_data, predictions, col_user="user", col_item="item", col_rating="playcount", col_prediction="prediction",  k=k_value)
        map_score = map(self.truth_data, predictions, col_user="user", col_item="item", col_rating="playcount", col_prediction="prediction",  k=k_value)

        return (precision_score, recall_score, ndcg_score, map_score)

    
def main():
    predictions = pd.DataFrame({
        'user': [1, 1, 2, 2, 3, 3],
        'item': [101, 102, 103, 107, 105, 108],
        'rank': [1, 2, 1, 2, 1, 2] 
    })
    

    content_evaluation = ContentEvaluation()

    test = content_evaluation.LenskitEvaluation(predictions)
    print("Lenskit Evaluation: ", test)
    
    test2 = content_evaluation.RecommenderEvaluation(predictions, 2)
    print("Recommender Evaluation: ", test2)

if __name__ == '__main__':
    main()
