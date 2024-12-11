import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.softmax import Softmax


class SoftmaxRecommender:
    def __init__(self, data: pd.DataFrame = None, meta_data: pd.DataFrame = None, user_data: pd.DataFrame = None):
        self.data = data
        self.meta_data = meta_data
        self.user_data = user_data
        self.model_path = "_softmax"
        self.tracks_data_with_metadata = pd.concat(
            [self.meta_data.reset_index(drop=True), self.data], axis=1
        )
        self.tracks_data = self.tracks_data_with_metadata.drop(columns=["name", "artist"])
        self.X_train, self.X_test, self.y_train, self.y_test = self.softmax_data()
        self.model = self.get_softmax()

    def softmax_data(self):
        # Splitting data into features and target
        features = self.tracks_data.drop(columns=['track_id'])
        target = self.tracks_data["track_id"]
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def get_softmax(self):
        if os.path.exists(f"{self.model_path}.keras"):
            model = Softmax(self.X_train, self.y_train, "")
            model.load_model(self.model_path)
        else:
            model = Softmax(self.X_train, self.y_train, "")
            model.train(epochs=10, batch_size=32)
            model.save(self.model_path)
        return model

    def recommend(self, user_profile, user_id: int, top_n: int):
        user_input = pd.DataFrame(user_profile)

        predictions = self.model.predict(user_input)

        indices = np.argsort(predictions[0])[::-1]

        if user_id:
            user_track = self.user_data[self.user_data['user_id'] == user_id]['track_id'].values
            length = top_n + len(user_track)
            reduced_indicies = indices[:length]
            filtered_indices = [item for item in reduced_indicies if item not in user_track]
        if top_n:
            final_indices = filtered_indices[:top_n]
        return final_indices, self.tracks_data_with_metadata.iloc[final_indices]

