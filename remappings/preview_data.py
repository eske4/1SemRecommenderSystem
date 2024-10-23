from pprint import pprint

import pandas as pd

# Step 1: Load the DataFrame using tab as the separator
print("Which dataset do you want to preview?")
print("Type 1 for user_id dictionary")
print("Type 2 for track_id dictionary")
print("Type 3 for tags dictionary")
print("Type 4 to preview music info remapped")
print("Type 5 to preview user listening history remapped")

# Read the datasets
user_ids_df = pd.read_csv("data/user_ids.txt", sep="\t")
track_ids_df = pd.read_csv("data/track_ids.txt", sep="\t")
tags_df = pd.read_csv("data/tags.txt", sep="\t")
modified_music_info_df = pd.read_csv("data/Modified_Music_info.txt", sep="\t")
modified_listening_history_df = pd.read_csv(
    "data/Modified_Listening_History.txt", sep="\t"
)


# Function to display DataFrame information
def display_dataframe(df, title):
    print(f"\n{title}:")
    print("Column headers:")
    pprint(df.columns.tolist())
    print("\nSample data:")

    # Set pandas display options for better alignment
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        # Use to_string() to present the DataFrame neatly without auto-indexing
        print(df.head().to_string(index=False))


# Mapping user input to functions
switch = {
    "1": lambda: display_dataframe(user_ids_df, "User ID Dictionary"),
    "2": lambda: display_dataframe(track_ids_df, "Track ID Dictionary"),
    "3": lambda: display_dataframe(tags_df, "Tags Dictionary"),
    "4": lambda: display_dataframe(modified_music_info_df, "Modified Music Info"),
    "5": lambda: display_dataframe(
        modified_listening_history_df, "Modified User Listening History"
    ),
}

# Step 2: Get user input
choice = input("Enter your choice (1-5): ")

# Step 3: Call the corresponding function or handle invalid input
switch.get(
    choice, lambda: print("Invalid choice. Please enter a number between 1 and 5.")
)()
