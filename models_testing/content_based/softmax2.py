import tensorflow as tf
import os
import pandas as pd
import numpy as np

class Softmax:
    def __init__(self, shape, meta_data: pd.DataFrame, model_path="models_testing/content_based/softmax.keras"):
        self.shape = shape
        self.num_unique_tracks = meta_data["track_id"].nunique()
        self.model_path = model_path  
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_unique_tracks, activation="softmax"),
        ])

    def train(self, X_train, y_train):
        if self.isTrained: 
            print("Is already trained")
            return
        if X_train is None or y_train is None:
            print("Train data is missing either x or y train")
            return
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Loaded pre-trained model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print(f"Model not found at {self.model_path}")
            self.model = None

    def recommend(self, user_history: pd.DataFrame, num_recommendations: int) -> tuple:
        user_id = user_history.iloc[0, 0]
        user_input = user_history.iloc[0, 1:].values.reshape(1, -1)
        user_input_scaled = self.scaler.transform(user_input)

        predictions = self.model.predict(user_input_scaled)

        top_n_indices = np.argsort(predictions[0])[-num_recommendations:][::-1]

        return user_id, top_n_indices
    
    def get_all_recommendations(self, average_feature_df: pd.DataFrame) -> pd.DataFrame:
        all_recommendations = []

        for i in range(len(average_feature_df)): 
            print("step", i, "out of", len(average_feature_df))
            user_history = average_feature_df.iloc[[i]]
            num_recommendations = 10
            user_id, top_n_indices = self.recommend(user_history, num_recommendations)
            
            user_ranks = list(range(1, num_recommendations + 1))
            song_ids = top_n_indices 
            
            for song, rank in zip(song_ids, user_ranks):
                all_recommendations.append({
                    'item': song,
                    'user': user_id,      
                    'rank': rank     
                })
        
        return pd.DataFrame(all_recommendations)

    def save(self, path):
        self.model.save(path)

    def call(self, x):
        return self.model(x)
