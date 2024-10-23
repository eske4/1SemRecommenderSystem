import pandas as pd


def load_csv_files():
    """Load CSV files into pandas DataFrames."""
    df = pd.read_csv("../data/User Listening History.csv")
    df2 = pd.read_csv("../data/Music Info.csv")
    return df, df2


def load_tag_dictionary(file_path):
    """Load tags from a text file into a dictionary."""
    tags_to_index = {}
    with open(file_path, "r") as f:
        for line in f:
            tag, index = line.strip().split(": ")
            tags_to_index[tag] = int(index)
    return tags_to_index


def replace_tags_with_index(tag_array, tags_to_index):
    """Replace tags with their index numbers."""
    if pd.isna(tag_array):
        return []  # Return an empty list for NaN
    if isinstance(tag_array, str):
        tag_array = tag_array.split(", ")  # Split by comma and space
    return [
        tags_to_index.get(tag.strip(), None) for tag in tag_array
    ]  # Strip spaces and get index


def load_id_mappings(file_path):
    """Load ID mappings from a text file into a dictionary."""
    ids = {}
    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split(":")
            ids[key.strip()] = int(value.strip())  # Strip spaces and convert to int
    return ids


def map_ids(df, track_ids, user_ids):
    """Map track and user IDs in the DataFrame."""
    df["track_id"] = df["track_id"].map(track_ids)
    df["user_id"] = df["user_id"].map(user_ids)


def preprocess_music_info(df2, track_ids, tags_to_index):
    """Preprocess the second DataFrame for music info."""
    df2["track_id"] = df2["track_id"].map(track_ids)
    df2["tags"] = df2["tags"].apply(lambda x: replace_tags_with_index(x, tags_to_index))
    return df2.drop(columns=["spotify_preview_url", "spotify_id"], errors="ignore")


def save_dataframes_to_txt(df, df2):
    """Save modified DataFrames to text files."""
    output_file_path1 = "data/Modified_Listening_History.txt"
    output_file_path2 = "data/Modified_Music_info.txt"

    df.to_csv(
        output_file_path1, sep="\t", index=False
    )  # Tab as separator for the text file
    df2.to_csv(
        output_file_path2, sep="\t", index=False
    )  # Tab as separator for the text file


def main():
    """Main function to execute the data processing steps."""
    # Step 1: Load data
    df, df2 = load_csv_files()

    # Step 2: Load mappings
    tags_to_index = load_tag_dictionary("data/tags.txt")
    track_ids = load_id_mappings("data/track_ids.txt")
    user_ids = load_id_mappings("data/user_ids.txt")

    # Step 3: Map IDs in the DataFrames
    map_ids(df, track_ids, user_ids)
    df2 = preprocess_music_info(df2, track_ids, tags_to_index)

    # Optional: Drop columns based on a condition
    with_genre = False
    if with_genre:
        df2 = df2.drop(
            columns=["tags"], errors="ignore"
        )  # Drop tags if with_genre is True
    else:
        df2 = df2.drop(
            columns=["genre"], errors="ignore"
        )  # Drop genre if with_genre is False

    # Step 4: Save the modified DataFrames to text files
    save_dataframes_to_txt(df, df2)

    print("Modified files have been saved successfully.")


if __name__ == "__main__":
    main()
