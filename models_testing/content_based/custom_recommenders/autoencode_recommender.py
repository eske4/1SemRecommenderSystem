import numpy as np
import pandas as pd
from models.autoencoder import Autoencoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


class AutoencodeRecommender:
    def __init__(self, data: pd.DataFrame = None, latent_dim: int = 90, meta_data: pd.DataFrame = None, user_data: pd.DataFrame = None, user_data_test: pd.DataFrame = None):
        self.data = data  # Store the input data
        self.meta_data = meta_data  # Store metadata
        self.latent_dim = latent_dim  # Store latent dimension
        self.user_data = user_data
        self.user_data_test = user_data_test
        self.encoded_data = self.autoencode_data()  # Encode data using autoencoder
        self.encoded_data_with_metadata = pd.concat(
            [self.meta_data.reset_index(drop=True), self.encoded_data], axis=1
        )

    def autoencode_data(self):
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )
        autoencoder = Autoencoder(self.latent_dim, train_data, "")
        autoencoder.validate(test_data)
        return pd.DataFrame(autoencoder.predict(self.data), columns=self.data.columns)

    def __sort_by_distance(self, input_feature, tracks_data, ):
        sim_scores = cosine_similarity([input_feature], tracks_data)[0]
        sorted_indices = np.argsort(sim_scores)[::-1]
        return sorted_indices

    def recommend_similar_items(self, input_feature, user_id, top_n=None):
        indices= self.__sort_by_distance(
            input_feature, self.encoded_data
        )
        if user_id:
            user_track = self.user_data[self.user_data['user_id'] == user_id]['track_id'].values
            length = top_n + len(user_track)
            reduced_indicies = indices[:length]
            track_ids_train = self.user_data['track_id'].unique()
            track_ids_test = self.user_data_test['track_id'].unique()
            combined_track_ids = np.unique(np.concatenate((track_ids_train,track_ids_test)))
            # Filter out items that are in the user's track list
            filtered_indices = [item for item in reduced_indicies if item not in user_track]
            # Further filter to include only items in track_ids_train
            filtered_indices = [item for item in filtered_indices if item in combined_track_ids]
        if top_n:
            final_indices = filtered_indices[:top_n]
        return final_indices, self.encoded_data_with_metadata.iloc[final_indices]
    
    

    def recommend_similar_items_within_range(
        self, min_score=0.5, max_score=0.7, input_feature=None, top_n=None
    ):
        indices, scores, items = self.__sort_by_distance(
            input_feature, self.encoded_data, self.encoded_data_with_metadata
        )
        range_mask = (scores > min_score) & (scores < max_score)
        indices = indices[range_mask]
        scores = scores[range_mask]
        items = items.iloc[range_mask]
        sort_by_score = np.argsort(scores)[::-1]
        if top_n:
            indices = indices[sort_by_score][:top_n]
            scores = scores[sort_by_score][:top_n]
            items = items.iloc[sort_by_score][:top_n]
        return indices, scores, items

    def recommend_by_diversity(
        self,
        isUsingRoom=False,
        min_score=0.5,
        max_score=0.7,
        input_feature=None,
        top_n=10,
    ):
        indices, scores, items = self.__sort_by_distance(
            input_feature, self.encoded_data, self.encoded_data_with_metadata
        )
        if isUsingRoom:
            indices, scores, items = self.recommend_similar_items_within_range(
                min_score, max_score, input_feature, top_n
            )

        recommended_indices = [indices[0]]
        recommended_scores = [scores[0]]
        recommended_items = [items.iloc[0]]

        for _ in range(1, top_n):
            max_dissimilarity_scores = []
            for idx, score in enumerate(scores):
                if idx in recommended_indices:
                    max_dissimilarity_scores.append(
                        -np.inf
                    )  # Exclude already recommended
                    continue
                current_item_feature = self.encoded_data.iloc[idx]
                dissimilarity_scores = [
                    cosine_similarity(
                        [current_item_feature], [self.encoded_data.iloc[rec_idx]]
                    )[0][0]
                    for rec_idx in recommended_indices
                ]
                max_dissimilarity_scores.append(
                    -max(dissimilarity_scores)
                )  # Negative for max dissimilarity

            next_index = np.argmax(max_dissimilarity_scores)
            recommended_indices.append(next_index)
            recommended_scores.append(scores[next_index])
            recommended_items.append(items.iloc[next_index])

            # Stop if we've reached the desired top_n
            if len(recommended_indices) >= top_n:
                break

        return (
            np.array(recommended_indices),
            np.array(recommended_scores),
            pd.DataFrame(recommended_items),
        )
