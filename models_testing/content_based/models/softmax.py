import numpy as np
import tensorflow as tf
from tensorflow.keras import Model



class Softmax(Model):
    def __init__(self, X_train=None, y_train=None, path=None):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.num_unique_tracks = len(X_train) + len(y_train)
        self.shape = self.X_train.shape[1:] if self.X_train is not None else None
        self.softmax = None
        if path is not None:
            self.load_model(path)
        elif self.shape:
            self._build_models()

    def _build_models(self):
        # Build the Softmax model
        self.softmax = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_unique_tracks, activation="softmax"),
        ])

    def train(self, epochs: int, batch_size: int):
        if not self.softmax:
            self._build_models()
        print("Training model...")
        self.softmax.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.softmax.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def save(self, path):
        # Save the model to the specified path
        self.softmax.save(f"{path}.keras")
        print(f"Model saved to {path}.keras")

    def load_model(self, path):
        try:
            # Load the model from the specified path
            self.softmax = tf.keras.models.load_model(f"{path}.keras")
            print(f"Loaded pre-trained model from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.softmax = None
        return self

    def predict(self, x):
        x = np.array(x).reshape(1, -1)  
        return self.softmax.predict(x)