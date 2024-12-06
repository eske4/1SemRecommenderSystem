import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

class GetUserHistory:
    def __init__(self) -> None:
        self.mlb =MultiLabelBinarizer()

    @staticmethod
    def parse_tags(x):
        if isinstance(x, str):
            if x.startswith("[") and x.endswith("]"):
                return eval(x)  # Risky; replace with ast.literal_eval for safer usage
            return [tag.strip() for tag in x.split(",") if tag.strip()]
        return []

    def prepare_musicdata(self, music_dataset: pd.DataFrame) -> pd.DataFrame:
        music_dataset["tags"] = music_dataset["tags"].apply(self.parse_tags)

        # Binarize tags
        tags_binarized = self.mlb.fit_transform(music_dataset['tags'])
        binary_tags = pd.DataFrame(tags_binarized, columns=self.mlb.classes_.astype(str))
        music_dataset = pd.concat([music_dataset, binary_tags], axis=1)

        excluded_columns = ['name', 'artist', 'tags', 'year', 'time_signature', 'duration_ms', 'genre']
        prepared_music_dataset = music_dataset.drop(columns=excluded_columns)

        return prepared_music_dataset

    def merge_dataset(self, user_dataset: pd.DataFrame, music_dataset: pd.DataFrame, exclude_columns: list = None) -> pd.DataFrame:
        if exclude_columns:
            music_dataset = music_dataset.drop(columns=exclude_columns)

        merged_dataset = pd.merge(user_dataset, music_dataset, on="track_id", how="left")

        return merged_dataset

    def get_average_features(self, merged_dataset: pd.DataFrame) -> pd.DataFrame:
        average_dataset = merged_dataset.groupby('user_id').mean()

        average_dataset.reset_index(inplace=True)

        exclude_columns = ['track_id', 'playcount']

        average_dataset = average_dataset.drop(columns=exclude_columns)

        return average_dataset
    
    def get_test_history(self, test_dataset: pd.DataFrame) -> pd.DataFrame:
        test_user_history = test_dataset.groupby('user_id')['track_id'].apply(list).reset_index()
        test_user_history.rename(columns={'track_id': 'track_ids'}, inplace=True)
        return test_user_history
    
def main():
    test_dataset = pd.read_csv('remappings/data/dataset/test_listening_history_OverEqual_50_Interactions.txt', delimiter='\t')
    train_dataset = pd.read_csv('remappings/data/dataset/train_listening_history_OverEqual_50_Interactions.txt', delimiter='\t')
    music_dataset = pd.read_csv('remappings/data/Modified_Music_info.txt', delimiter='\t')
    
    ranking = GetUserHistory()

    binarized_music_dataset = ranking.prepare_musicdata(music_dataset)   

    merged_dataset = ranking.merge_dataset(train_dataset, binarized_music_dataset)

    print(merged_dataset)

    average_features_dataset = ranking.get_average_features(merged_dataset)

    print(average_features_dataset)

    group_test = ranking.get_test_history(test_dataset)

    print(group_test)
if __name__ == "__main__":
    main()