import numpy as np 
import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

#load data
file_path = r"remappings/data/Modified_Music_info.txt"
music_data = pd.read_csv(file_path, delimiter="\t")

def parse_tags(x):
    if isinstance(x, str):
        # Check if the string looks like a list
        if x.startswith("[") and x.endswith("]"):
            # Convert from string representation of a list to a list
            return eval(
                x
            )  # This could still be risky, consider using ast.literal_eval
        else:
            # Split by commas and strip whitespace
            return [tag.strip() for tag in x.split(",") if tag.strip()]
    return []

music_data["tags"] = music_data["tags"].apply(parse_tags)


mlb = MultiLabelBinarizer()
tags_binarized = mlb.fit_transform(music_data['tags'])
binary_tags = pd.DataFrame(tags_binarized, columns=mlb.classes_.astype(str))


music_data = pd.concat([music_data, binary_tags], axis=1)

print(music_data.head())

#define features and target
# note: easier to drop not used colloms 
selected_features = ["danceability", "energy","key", "loudness", "mode", "speechiness","acousticness", "instrumentalness", "liveness", "valence", "tempo",] + list(binary_tags.columns)
features = music_data[selected_features]
target = music_data["name"]

# data splitting
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_unique_tracks = (target.max() + 1) 

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_unique_tracks, activation="softmax"),
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# training parameters
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

#find track parameters based on song name id 
user_history = [0] #insert name id for user id
user_history_all_parameters = music_data[music_data["name"].isin(user_history)]
user_history_selected_parameters = user_history_all_parameters[selected_features]
user_history_selected_parameters_array = user_history_selected_parameters.values.tolist()

# example user parameters for user song history 
user_input = np.array(user_history_selected_parameters_array)
user_input_scaled = scaler.transform(user_input)

# number of songs wanted to be recommended to user
num_recommendations = 10

predictions = model.predict(user_input_scaled)

top_n_indices = np.argsort(predictions[0])[-num_recommendations:][::-1]
top_n_songs = music_data.iloc[top_n_indices]['name']

print(top_n_songs)