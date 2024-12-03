import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from softmax_get_user_history import SoftmaxGetUserHistory
from lenskit import topn



class SoftmaxRecommender:
    def __init__(self, file_path, model_path="models_testing/activation_functions/softmax.keras"):
        self.file_path = file_path
        self.model_path = model_path
        self.music_data = None
        self.mlb = None
        self.scaler = None
        self.model = None
        self.selected_features = None
        self.num_unique_tracks = None

    @staticmethod
    def parse_tags(x):
        if isinstance(x, str):
            if x.startswith("[") and x.endswith("]"):
                return eval(x)  # Risky; replace with ast.literal_eval for safer usage
            return [tag.strip() for tag in x.split(",") if tag.strip()]
        return []

    def load_and_prepare_data(self):
        self.music_data = pd.read_csv(self.file_path, delimiter="\t")

        self.music_data["tags"] = self.music_data["tags"].apply(self.parse_tags)

        self.mlb = MultiLabelBinarizer()
        tags_binarized = self.mlb.fit_transform(self.music_data['tags'])
        binary_tags = pd.DataFrame(tags_binarized, columns=self.mlb.classes_.astype(str))
        self.music_data = pd.concat([self.music_data, binary_tags], axis=1)

        self.selected_features = [
            "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo"
        ] + list(binary_tags.columns)
        
        self.num_unique_tracks = self.music_data["track_id"].nunique()

    def preprocess_data(self):
        features = self.music_data[self.selected_features]
        target = self.music_data["track_id"]
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train.values)
        X_test = self.scaler.transform(X_test.values)

        return X_train, X_test, y_train, y_test

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

    def build_and_train_model(self, X_train, y_train):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_unique_tracks, activation="softmax"),
        ])

        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        self.model.save(self.model_path)

    def recommend(self, user_history, num_recommendations=10):
        user_id = np.array(user_history[0][0])
        user_input = np.array([user_history[0][1:]])
        user_input_scaled = self.scaler.transform(user_input)

        predictions = self.model.predict(user_input_scaled)

        top_n_indices = np.argsort(predictions[0])[-num_recommendations:][::-1]
        top_n_predictions_scores = np.sort(predictions[0])[-num_recommendations:][::-1]

        return user_id, top_n_indices


def main():
    file_path = "remappings/data/Modified_Music_info.txt"

    recommender = SoftmaxRecommender(file_path)

    recommender.load_and_prepare_data()

    X_train, X_test, y_train, y_test = recommender.preprocess_data()

    recommender.load_model()

    if recommender.model is None:
        recommender.build_and_train_model(X_train, y_train)

    test_dataset = pd.read_csv('remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt', delimiter='\t')
    train_dataset = pd.read_csv('remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt', delimiter='\t')
    music_dataset = pd.read_csv('remappings/data/Modified_Music_info.txt', delimiter='\t')
    
    ranking = SoftmaxGetUserHistory()

    binarized_music_dataset = ranking.prepare_musicdata(music_dataset)   

    merged_dataset = ranking.merge_dataset(train_dataset, binarized_music_dataset)

    average_features_dataset = ranking.get_average_features(merged_dataset)

    all_recommendations = []

    for i in range(len(average_features_dataset)): 
        user_history = [average_features_dataset.iloc[i]]
        num_recommendations = 10
        user_id, top_n_indices = recommender.recommend(user_history, num_recommendations)
        
        user_ranks = list(range(1, num_recommendations + 1))
        song_ids = top_n_indices 
        
        for song, rank in zip(song_ids, user_ranks):
            all_recommendations.append({
                'item': song,
                'user': user_id,      
                'rank': rank     
            })
        print("step", i, "out of", len(average_features_dataset))

    df_all_recommendations = pd.DataFrame(all_recommendations)

    
    test_dataset.rename(columns={'track_id': 'item', 'user_id': 'user'}, inplace=True)

    truth = test_dataset.drop(columns='playcount')
    predicted = df_all_recommendations
    truth['user'] = truth['user'].astype(int)
    predicted['user'] = predicted['user'].astype(int)

    predicted_test = pd.DataFrame({
        'item': [29637, 11164, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3803, 4, 5, 6, 7, 8, 9, 10], 
        'user': [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        'rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })

    print("truth: ", truth, "predicted: ", predicted)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    results = rla.compute(predicted, truth)
    precision_at_k = results['precision'].mean()
    recall_at_k = results['recall'].mean()

    print(f'Precision@k: {precision_at_k:.10f}')
    print(f'Recall@k: {recall_at_k:.10f}')

if __name__ == "__main__":
    main()
