import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


class Autoencoder(Model):
    def __init__(self, latent_dim=None, data=None, path=None):
        super().__init__()
        self.data, self.latent_dim = data, latent_dim
        self.shape = data.shape[1:] if data is not None else None
        if path is not None:
            self.load(path)
        elif self.shape:
            self._build_models()

    def _build_models(self):
        self.encoder = tf.keras.Sequential(
            [layers.Flatten(), layers.Dense(self.latent_dim, "relu")]
        )
        self.decoder = tf.keras.Sequential(
            [layers.Dense(np.prod(self.shape), "sigmoid"), layers.Reshape(self.shape)]
        )
        self.autoencoder = tf.keras.Sequential([self.encoder, self.decoder])

    def train(self, epochs=30, batch_size=512):
        if not self.data:
            raise ValueError("Data required.")
        if not self.autoencoder:
            self._build_models()
        self.autoencoder.compile("adam", "mean_squared_error")
        self.autoencoder.fit(self.data, self.data, epochs=epochs, batch_size=batch_size)

    def save(self, path):
        self.autoencoder.save(f"{path}_autoencoder.keras")

    def load(self, path):
        self.autoencoder = tf.keras.models.load_model(f"{path}_autoencoder.keras")
        self.encoder, self.decoder = self.autoencoder.layers[:2]
        self.shape, self.latent_dim = (
            self.encoder.input_shape[1:],
            self.encoder.layers[-1].units,
        )
        return self

    def call(self, x):
        return self.decoder(self.encoder(x))

    def predict(self, x):
        return self.autoencoder.predict(x)

    def validate(self, test_data):
        """
        Validates the autoencoder on test data by calculating reconstruction loss.
        """
        # Compute the reconstruction loss
        loss = self.autoencoder.evaluate(test_data, test_data, verbose=2)
        print(f"Reconstruction Loss: {loss}")

        # Compute reconstruction errors for each sample
        reconstructed = self.autoencoder.predict(test_data)
        errors = tf.reduce_mean(
            tf.square(test_data - reconstructed), axis=tuple(range(1, test_data.ndim))
        )

        return loss, errors
