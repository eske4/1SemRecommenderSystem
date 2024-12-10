import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#Base case without tensorflow model: recommends only on cosinesimilarity to 

class CosineRecommender:
    def __init__(self, data: pd.DataFrame = None , meta_data: pd.DataFrame = None, user_data: pd.DataFrame = None):
        self.tracks_data = data
        self.meta_data = meta_data
        self.user_data = user_data
        self.tracks_data_with_metadata = pd.concat(
            [self.meta_data.reset_index(drop=True), self.tracks_data], axis=1
        )
        
    def __sort_by_distance(self, input_feature, tracks_data, ):
        sim_scores = cosine_similarity([input_feature], tracks_data)[0]
        sorted_indices = np.argsort(sim_scores)[::-1]
        return sorted_indices

    def recommend_similar_items(self, input_feature, user_id, top_n=None):
        indices= self.__sort_by_distance(
            input_feature, self.tracks_data
        )
        if user_id:
            user_track = self.user_data[self.user_data['user_id'] == user_id]['track_id'].values
            print(indices)
            indices = [index for index in indices if index not in user_track]
        if top_n:
            indices = indices[:top_n]
        return indices, self.tracks_data_with_metadata.iloc[indices]
