"""LSTM Autoencoder for temporal anomaly detection in GridSense AI.

TensorFlow is imported lazily to avoid slow startup (~3-8s + 400MB RAM)
when this module is imported by other layers.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

INPUT_TIMESTEPS = 48
N_FEATURES = 7
LATENT_DIM = 32


class GridSenseLSTM:
    """LSTM Autoencoder for sequence-level anomaly detection on transformer readings.

    Architecture:
        Input: (48, 7)
        LSTM(32, return_sequences=False)
        RepeatVector(48)
        LSTM(32, return_sequences=True)
        TimeDistributed(Dense(7))
        compile: optimizer=adam, loss=mae

    High reconstruction error on a sequence = temporal anomaly.
    TensorFlow imported lazily to keep other module imports fast.
    """

    def __init__(self) -> None:
        """Initialize the LSTM Autoencoder without importing TensorFlow."""
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._threshold_99: float = 1.0  # 99th-pct reconstruction error on training data
        self._is_fitted: bool = False

    def _build_model(self) -> Any:
        """Build and return the Keras LSTM autoencoder architecture."""
        from tensorflow import keras  # lazy import

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(INPUT_TIMESTEPS, N_FEATURES)),
                keras.layers.LSTM(LATENT_DIM, return_sequences=False),
                keras.layers.RepeatVector(INPUT_TIMESTEPS),
                keras.layers.LSTM(LATENT_DIM, return_sequences=True),
                keras.layers.TimeDistributed(keras.layers.Dense(N_FEATURES)),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
        return model

    def train(self, X: np.ndarray, epochs: int = 20, batch_size: int = 32) -> Any:
        """Train the autoencoder on sequences of normal transformer readings.

        Args:
            X: array of shape (n_sequences, 48, 7) containing normal sequences
            epochs: number of training epochs
            batch_size: mini-batch size
        Returns:
            Keras training History object
        """
        from sklearn.preprocessing import StandardScaler  # lazy import

        n, t, f = X.shape
        # Fit scaler on flattened (n*t, f), then reshape back
        X_flat = X.reshape(n * t, f)
        self._scaler = StandardScaler()
        X_scaled_flat = self._scaler.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n, t, f)

        self._model = self._build_model()
        history = self._model.fit(
            X_scaled,
            X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        # Compute 99th-percentile threshold from training reconstruction errors
        X_pred = self._model.predict(X_scaled, verbose=0)
        errors = np.mean(np.abs(X_pred - X_scaled), axis=(1, 2))
        self._threshold_99 = float(np.percentile(errors, 99))
        self._is_fitted = True
        logger.info(
            "LSTM Autoencoder trained: epochs=%d, n_sequences=%d, threshold_99=%.4f",
            epochs,
            n,
            self._threshold_99,
        )
        return history

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized reconstruction error for input sequences.

        Args:
            X: array of shape (n_sequences, 48, 7)
        Returns:
            errors: array of shape (n_sequences,) with values in [0.0, 1.0]
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        n, t, f = X.shape
        X_flat = X.reshape(n * t, f)
        X_scaled_flat = self._scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n, t, f)

        X_pred = self._model.predict(X_scaled, verbose=0)
        errors = np.mean(np.abs(X_pred - X_scaled), axis=(1, 2))
        # Normalize by 99th-pct threshold; clip to [0, 1]
        normalized = errors / max(self._threshold_99, 1e-8)
        return np.clip(normalized, 0.0, 1.0)

    def save(self, path: str) -> None:
        """Save Keras model and scaler/threshold metadata to disk.

        Creates two files: ``{path}.keras`` and ``{path}_meta.joblib``.

        Args:
            path: base path without extension (e.g., 'models/saved/lstm_autoencoder')
        """
        import joblib  # lazy import

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._model.save(path + ".keras")
        joblib.dump(
            {
                "scaler": self._scaler,
                "threshold_99": self._threshold_99,
                "is_fitted": self._is_fitted,
            },
            path + "_meta.joblib",
        )
        logger.info("LSTM autoencoder saved to %s", path)

    def load(self, path: str) -> None:
        """Load Keras model and metadata from disk.

        Args:
            path: same base path used in :meth:`save`
        """
        import joblib
        from tensorflow import keras

        self._model = keras.models.load_model(path + ".keras")
        meta = joblib.load(path + "_meta.joblib")
        self._scaler = meta["scaler"]
        self._threshold_99 = meta["threshold_99"]
        self._is_fitted = meta["is_fitted"]
        logger.info("LSTM autoencoder loaded from %s", path)

    @classmethod
    def from_file(cls, path: str) -> "GridSenseLSTM":
        """Load a pre-trained LSTM model from disk.

        Args:
            path: base path (without extension)
        Returns:
            Fitted GridSenseLSTM instance
        """
        instance = cls()
        instance.load(path)
        return instance
