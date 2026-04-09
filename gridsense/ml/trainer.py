"""GridSense AI ML model trainer.

Generates synthetic data, trains Isolation Forest and LSTM Autoencoder,
and saves artifacts to gridsense/ml/models/saved/.

Run once before starting the main system:
    uv run python -m gridsense.ml.trainer
"""
from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved")
IF_MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.joblib")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_autoencoder")


def train_isolation_forest() -> None:
    """Generate synthetic data and train the Isolation Forest model."""
    import numpy as np

    from gridsense.ml.isolation_forest import GridSenseIsolationForest
    from gridsense.ml.utils.data_loader import (
        generate_anomalous_readings,
        generate_normal_readings,
    )

    logger.info("Generating training data for Isolation Forest…")
    X_normal = generate_normal_readings(n_samples=10_000)
    X_anomaly = generate_anomalous_readings(n_samples=500)
    X_train = np.vstack([X_normal, X_anomaly])

    logger.info("Training Isolation Forest on %d samples…", len(X_train))
    model = GridSenseIsolationForest(contamination=0.05, n_estimators=100)
    model.fit(X_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(IF_MODEL_PATH)

    # Quick validation
    test_normal = generate_normal_readings(n_samples=100, random_seed=1)
    test_anomaly = generate_anomalous_readings(n_samples=100, random_seed=2)
    normal_scores = model.score(test_normal)
    anomaly_scores = model.score(test_anomaly)
    logger.info(
        "IF validation — Normal avg: %.4f | Anomaly avg: %.4f",
        normal_scores.mean(),
        anomaly_scores.mean(),
    )


def train_lstm() -> None:
    """Generate sequence data and train the LSTM Autoencoder."""
    import numpy as np

    from gridsense.ml.lstm_autoencoder import INPUT_TIMESTEPS, GridSenseLSTM
    from gridsense.ml.utils.data_loader import (
        extract_lstm_features,
        generate_anomalous_readings,
        generate_normal_readings,
        make_sequences,
    )

    logger.info("Generating training sequences for LSTM Autoencoder…")
    X_raw = generate_normal_readings(n_samples=5_000)
    X_features = extract_lstm_features(X_raw)         # (5000, 7)
    X_seq = make_sequences(X_features, timesteps=INPUT_TIMESTEPS)  # (n, 48, 7)

    if len(X_seq) == 0:
        logger.error("Not enough data to form LSTM sequences. Skipping LSTM training.")
        return

    logger.info("Training LSTM on %d sequences (shape %s)…", len(X_seq), X_seq.shape)
    model = GridSenseLSTM()
    model.train(X_seq, epochs=20, batch_size=32)
    model.save(LSTM_MODEL_PATH)

    # Validation: anomalous sequences should produce higher reconstruction error
    X_anom_raw = generate_anomalous_readings(n_samples=200, random_seed=55)
    X_anom_feat = extract_lstm_features(X_anom_raw)
    X_anom_seq = make_sequences(X_anom_feat, timesteps=INPUT_TIMESTEPS)
    if len(X_anom_seq) > 0:
        norm_errors = model.reconstruction_error(X_seq[:200])
        anom_errors = model.reconstruction_error(X_anom_seq[:200])
        logger.info(
            "LSTM validation — Normal error: %.4f | Anomaly error: %.4f",
            norm_errors.mean(),
            anom_errors.mean(),
        )


def main() -> None:
    """Train all GridSense AI ML models and print a summary."""
    print("=" * 58)
    print("  GridSense AI — ML Model Training")
    print("=" * 58)

    train_isolation_forest()
    train_lstm()

    print("=" * 58)
    print("  Training complete! Artifacts saved to:")
    print(f"  {MODELS_DIR}")
    print("    isolation_forest.joblib")
    print("    lstm_autoencoder.keras")
    print("    lstm_autoencoder_meta.joblib")
    print("=" * 58)


if __name__ == "__main__":
    main()
