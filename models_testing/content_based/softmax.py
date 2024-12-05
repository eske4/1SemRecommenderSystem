import tensorflow as tf

class Softmax:
    def __init__ (self, shape):
        self.shape = shape
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_unique_tracks, activation="softmax"),
        ])

        self.isTrained = False

    def train(self, X_train, y_train):
        if(self.isTrained == True): 
            print("Is already trained")
            return
        if(X_train == None or y_train == None):
            print("Train data is missing either x or y train")
            return
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    def save(self, path):
        self.model.save(path)

    def call(self, x):
        return self.model(x)

