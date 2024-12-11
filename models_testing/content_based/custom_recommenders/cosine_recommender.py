import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#Base case without tensorflow model: recommends only on cosinesimilarity to 

class CosineRecommender:
    def __init__(self, data: pd.DataFrame = None , meta_data: pd.DataFrame = None, user_data: pd.DataFrame = None):
        self.data = data
        self.meta_data = meta_data
        self.user_data = user_data
        self.tracks_data_with_metadata = pd.concat(
            [self.meta_data.reset_index(drop=True), self.data], axis=1
        )
        
    def __sort_by_distance(self, input_feature, tracks_data, ):
        sim_scores = cosine_similarity([input_feature], tracks_data)[0]
        sorted_indices = np.argsort(sim_scores)[::-1]
        return sorted_indices

    def recommend_similar_items(self, input_feature, user_id, top_n=None):
        indices= self.__sort_by_distance(
            input_feature, self.data
        )
        if user_id:
            user_track = self.user_data[self.user_data['user_id'] == user_id]['track_id'].values
            length = top_n + len(user_track)
            reduced_indicies = indices[:length]
            filtered_indices = [item for item in reduced_indicies if item not in user_track]
        if top_n:
            final_indices = filtered_indices[:top_n]
        return final_indices, self.tracks_data_with_metadata.iloc[final_indices]
