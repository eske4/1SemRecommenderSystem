import pandas as pd

# Step 1: Load the DataFrames
user_listening_history_df = pd.read_csv("../data/User Listening History.csv")
music_info_df = pd.read_csv("../data/Music Info.csv")

# Step 2: Get unique user IDs and track IDs
unique_user_ids = user_listening_history_df["user_id"].unique()
unique_track_ids = music_info_df["track_id"].unique()
unique_names = music_info_df["name"].unique()
unique_artists = music_info_df["artist"].unique()
unique_genres = music_info_df["genre"].unique()

# Create dictionaries to map user IDs and track IDs to their corresponding indices
user_id_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids)}
track_id_to_index = {track_id: index for index, track_id in enumerate(unique_track_ids)}
track_names_to_index = {name: index for index, name in enumerate(unique_names)}
track_artists_to_index = {artist: index for index, artist in enumerate(unique_artists)}
track_genre_to_index = {genre: index for index, genre in enumerate(unique_genres)}

# Step 3: Extract unique tags from the music_info_df
unique_tags = set()

# Assuming 'tags' is a column in music_info_df that contains lists or comma-separated strings of tags
for tags in music_info_df["tags"].dropna():  # Drop NaN values
    tag_list = [
        tag.strip() for tag in tags.split(",")
    ]  # Split by comma and strip whitespace
    unique_tags.update(tag_list)  # Update the set with unique tags

# Create a dictionary to map tags to their corresponding indices
tag_to_index = {tag: index for index, tag in enumerate(unique_tags)}

# Step 4: Write user ID and track ID dictionaries to text files with headers
with open("data/user_ids.txt", "w", encoding="utf-8") as user_file:
    for user_id, index in user_id_to_index.items():
        user_file.write(f"{user_id}\t {index}\n")

with open("data/track_ids.txt", "w", encoding="utf-8") as track_file:
    for track_id, index in track_id_to_index.items():
        track_file.write(f"{track_id}\t {index}\n")

# Step 5: Write the tag dictionary to a text file with a header
with open("data/tags.txt", "w", encoding="utf-8") as tag_file:
    for tag, index in tag_to_index.items():
        tag_file.write(f"{tag}\t {index}\n")

with open("data/genres.txt", "w", encoding="utf-8") as tag_file:
    for genre, index in track_genre_to_index.items():
        tag_file.write(f"{genre}\t {index}\n")

with open("data/artists.txt", "w", encoding="utf-8") as tag_file:
    for artist, index in track_artists_to_index.items():
        tag_file.write(f"{artist}\t {index}\n")

with open("data/names.txt", "w", encoding="utf-8") as tag_file:
    for name, index in track_names_to_index.items():
        tag_file.write(f"{name}\t {index}\n")

# Print confirmation messages
print("Dictionary files created")
