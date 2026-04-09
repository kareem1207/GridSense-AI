"""Isolation Forest anomaly detector for GridSense transformer monitoring."""
from __future__ import annotations
import logging
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GridSenseIsolationForest:
    """Isolation Forest wrapper for point anomaly detection on transformer readings.

    Features (11): [Va, Vb, Vc, Ia, Ib, Ic, oil_temp, power_factor, thd_pct, active_power_kw, reactive_power_kvar]
    Score range: 0.0 (normal) to 1.0 (highly anomalous).
    """

    FEATURE_NAMES = [
        "Va", "Vb", "Vc", "Ia", "Ib", "Ic",
        "oil_temp", "power_factor", "thd_pct",
        "active_power_kw", "reactive_power_kvar",
    ]

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        """Initialize the Isolation Forest.

        Args:
            contamination: expected fraction of anomalies in training data
            n_estimators: number of trees in the forest
            random_state: reproducibility seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Fit the Isolation Forest on training data.

        Args:
            X: array of shape (n_samples, 11) with DTM features
        """
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        self._is_fitted = True
        logger.info("IsolationForest fitted on %d samples", len(X))

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores normalized to [0, 1].

        Formula: score = clip((1 - decision_function(X)) / 2, 0, 1)
        Higher score means more anomalous.

        Args:
            X: array of shape (n_samples, 11)
        Returns:
            scores: array of shape (n_samples,) in [0.0, 1.0]
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self._scaler.transform(X)
        raw = self._model.decision_function(X_scaled)
        scores = (1.0 - raw) / 2.0
        return np.clip(scores, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal based on score > 0.5."""
        return (self.score(X) > 0.5).astype(int)

    def save(self, path: str) -> None:
        """Save model and scaler to disk using joblib.

        Args:
            path: file path (e.g., 'models/saved/isolation_forest.joblib')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {"model": self._model, "scaler": self._scaler, "is_fitted": self._is_fitted},
            path,
        )
        logger.info("IsolationForest saved to %s", path)

    def load(self, path: str) -> None:
        """Load model and scaler from disk.

        Args:
            path: path to the saved .joblib file
        """
        data = joblib.load(path)
        self._model = data["model"]
        self._scaler = data["scaler"]
        self._is_fitted = data["is_fitted"]
        logger.info("IsolationForest loaded from %s", path)

    @classmethod
    def from_file(cls, path: str) -> "GridSenseIsolationForest":
        """Load a pre-trained model from disk.

        Args:
            path: path to saved .joblib file
        Returns:
            Fitted GridSenseIsolationForest instance
        """
        instance = cls()
        instance.load(path)
        return instance
