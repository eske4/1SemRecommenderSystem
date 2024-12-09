import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.softmax import Softmax


class SoftmaxRecommender:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self.model_path = "_softmax"
        self.X_train, self.X_test, self.y_train, self.y_test = self.softmax_data()
        self.model = self.get_softmax()

    def softmax_data(self):
        # Splitting data into features and target
        features = self.data.drop(columns=['track_id'])
        target = self.data["track_id"]
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

    def recommend(self, user_profile, k: int):
        user_input = pd.DataFrame(user_profile)

        predictions = self.model.predict(user_input)

        top_k_indices = np.argsort(predictions[0])[-k:][::-1]

        return top_k_indices

