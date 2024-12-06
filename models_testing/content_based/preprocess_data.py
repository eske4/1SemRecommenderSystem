import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer


class PreprocessData:
    def __init__(self):
        """
        Initialize the processor with file paths.
        """
        self.track_path = "../../remappings/data/Modified_Music_info.txt"
        self.ratings_path = "../../remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt"
        self.scaler = None

    def load_track_features(self, path, max_rows=None):
        """
        Load track features from a specified file.
        """
        return pd.read_csv(path, delimiter="\t", nrows=max_rows)

    def prepare_data(self, data):
        """
        Normalize the data using MinMaxScaler.
        """
        self.scaler = MinMaxScaler()
        data_normalized = self.scaler.fit_transform(data)
        return data_normalized

    def preprocess_features(self, data):
        """
        Preprocess features by encoding categorical data and handling tags.
        """
        metadata = data[["name", "artist", "track_id"]]
        features = data.drop(columns=["name", "artist", "track_id"], errors="ignore")

        # Encode 'genre' as categorical
        if "genre" in features.columns:
            features["genre"] = features["genre"].astype("category")

        # Process 'tags' column
        if "tags" in features.columns:
            features["tags"] = features["tags"].apply(
                lambda x: (
                    eval(x)
                    if isinstance(x, str) and x.startswith("[") and x.endswith("]")
                    else (
                        [tag.strip() for tag in x.split(",") if tag.strip()]
                        if isinstance(x, str)
                        else []
                    )
                )
            )

            # One-hot encode tags
            mlb = MultiLabelBinarizer()
            tags_encoded = pd.DataFrame(
                mlb.fit_transform(features["tags"]), columns=mlb.classes_
            ).add_prefix("tag_")
        else:
            tags_encoded = pd.DataFrame()

        # One-hot encode genre
        if "genre" in features.columns:
            genre_encoded = pd.get_dummies(features["genre"], prefix="genre")
        else:
            genre_encoded = pd.DataFrame()

        # Combine processed features
        processed_data = pd.concat([features, genre_encoded, tags_encoded], axis=1)
        return processed_data.drop(columns=["genre", "tags", "year"], errors="ignore"), metadata

    def process_all_data(self, max_rows=None):
        """
        Process the tracks and ratings data and return processed features and metadata.
        """
        # Load raw tracks
        raw_tracks = self.load_track_features(self.track_path, max_rows=max_rows)
        # Preprocess track features
        processed_tracks, metadata = self.preprocess_features(raw_tracks)

        # Load ratings
        raw_ratings = self.load_track_features(self.ratings_path, max_rows=max_rows)

        return processed_tracks, metadata, raw_ratings
