import numpy as np
import pandas as pd
from models.autoencoder import Autoencoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


class AutoencodeRecommender:
    def __init__(self, data=None, latent_dim=90, meta_data=None):
        self.data = data  # Store the input data
        self.meta_data = meta_data  # Store metadata
        self.latent_dim = latent_dim  # Store latent dimension
        self.encoded_data = self.autoencode_data()  # Encode data using autoencoder
        self.encoded_data_with_metadata = pd.concat(
            [self.meta_data.reset_index(drop=True), self.encoded_data], axis=1
        )

    def autoencode_data(self):
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )
        autoencoder = Autoencoder(self.latent_dim, train_data, "")
        return pd.DataFrame(autoencoder.predict(self.data), columns=self.data.columns)

    def __sort_by_distance(self, input_feature, encoded_data, items):
        sim_scores = cosine_similarity([input_feature], encoded_data)[0]
        sorted_indices = np.argsort(sim_scores)[::-1]
        sorted_scores = sim_scores[sorted_indices]
        sorted_items = items.iloc[sorted_indices]
        return sorted_indices, sorted_scores, sorted_items

    def recommend_similar_items(self, input_feature, top_n=None):
        indices, scores, items = self.__sort_by_distance(
            input_feature, self.encoded_data, self.encoded_data_with_metadata
        )
        if top_n:
            indices = indices[:top_n]
            scores = scores[:top_n]
            items = items.iloc[:top_n]
        return indices, scores, items

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
