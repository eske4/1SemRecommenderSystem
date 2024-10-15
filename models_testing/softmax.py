import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and filter the dataset
file_path = 'C:\csSEVEN\.venv\dataset\Music Info.csv'
music_data = pd.read_csv(file_path)

# Select relevant columns
filtered_music_data = music_data[['name', 'genre', 'danceability', 'energy', 'loudness']]

# Filter out songs with a genre tag
songs_with_genre = filtered_music_data[filtered_music_data['genre'].notna()]

# Prepare features and labels
X = songs_with_genre[['danceability', 'energy', 'loudness']].values
y = songs_with_genre['genre'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

#neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Prepare songs without genre for prediction
songs_without_genre = filtered_music_data[filtered_music_data['genre'].isna()]
X_predict = scaler.transform(songs_without_genre[['danceability', 'energy', 'loudness']].values)

# predict genres for songs without a genre tags
predicted_genres = model.predict(X_predict)
predicted_genres_labels = label_encoder.inverse_transform(np.argmax(predicted_genres, axis=1))

# Assign the predicted genres to the songs
songs_without_genre['predicted_genre'] = predicted_genres_labels


print(songs_without_genre[['name', 'predicted_genre']])

#plots for genre desstrebution 
genre_counts = pd.Series(y).value_counts()
genre_counts.plot(kind='bar')
plt.title("Genre Distribution")
plt.xlabel("Genres")
plt.ylabel("Count")
plt.show()

predicted_genre_counts = songs_without_genre['predicted_genre'].value_counts()
predicted_genre_counts.plot(kind='bar')
plt.title("Predicted Genre Distribution for Songs Without Genre")
plt.xlabel("Genres")
plt.ylabel("Count")
plt.show()
