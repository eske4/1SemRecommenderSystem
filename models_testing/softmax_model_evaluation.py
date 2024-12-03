import pandas as pd
from lenskit import topn

class ContentEvaluation:
    def __init__(self):
        self.truth_data = pd.read_csv('remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt', delimiter='\t')
        self.truth_data.rename(columns={'track_id': 'item', 'user_id': 'user'}, inplace=True)

    def LenskitEvaluation(self, predictions: pd.DataFrame) -> tuple:
        '''
        predictions = pd.DataFrame({
            'item': [Int], 
            'user': [Int],
            'rank': [Int],
        })
        '''
        truth = self.truth_data
        truth['user'] = truth['user'].astype(int)
        predictions['user'] = predictions['user'].astype(int)

        # Futurewarning at ..\lenskit\metrics\topn.py line 100 and 149. replace lines with "scores['ngood'] = scores['ngood'].fillna(0)"

        rla = topn.RecListAnalysis()
        rla.add_metric(topn.precision)
        rla.add_metric(topn.recall)
        rla.add_metric(topn.hit)
        
        results = rla.compute(predictions, truth)
        precision_at_k = results['precision'].mean()
        recall_at_k = results['recall'].mean()
        hit_ratio_at_k = results['hit'].mean()

        return precision_at_k, recall_at_k, hit_ratio_at_k

def main():
    content_evaluation = ContentEvaluation()

if __name__ == '__main__':
    main()
