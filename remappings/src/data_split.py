import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the datasets
listening_history = pd.read_csv('../data/Modified_Listening_History.txt', delimiter='\t')

# Step 1: Filter out users and tracks with only one interaction
user_counts = listening_history['user_id'].value_counts()
track_counts = listening_history['track_id'].value_counts()

# Keep only users and tracks that appear more than (whatever value you put in)
multiple_users = user_counts[user_counts >= 50].index
multiple_tracks = track_counts[track_counts > 1].index

filtered_listening_history = listening_history[
    (listening_history['user_id'].isin(multiple_users)) & 
    (listening_history['track_id'].isin(multiple_tracks))
]

# Function to split each user's interactions into train and test
def split_user_data(user_data):
    if len(user_data) > 1:
        train, test = train_test_split(user_data, test_size=0.2, random_state=42)
        return pd.DataFrame({"train": [train], "test": [test]})
    else:
        # Return the user data as train only if there's a single interaction
        return pd.DataFrame({"train": [user_data], "test": [pd.DataFrame()]})

# Step 2: Apply the function to each user group
split_data = filtered_listening_history.groupby('user_id').apply(split_user_data).reset_index(drop=True)

# Step 3: Concatenate all train and test sets into final DataFrames
train_data = pd.concat(split_data["train"].tolist(), ignore_index=True)
test_data = pd.concat(split_data["test"].tolist(), ignore_index=True)

os.chdir("../data")
os.mkdir("dataset")
os.chdir("dataset")
# Step 4: Save the Train and Test Split
train_data.to_csv('train_listening_history_OverEqual_50_Interactions.txt', sep='\t', index=False)
test_data.to_csv('test_listening_history_OverEqual_50_Interactions.txt', sep='\t', index=False)

print("Train and test splits saved successfully.")