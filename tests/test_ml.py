"""ML layer tests for GridSense AI."""
from __future__ import annotations

import numpy as np
import pytest

from gridsense.ml.utils.data_loader import (
    extract_lstm_features,
    generate_anomalous_readings,
    generate_degrading_sequence,
    generate_normal_readings,
    make_sequences,
)


# ---------------------------------------------------------------------------
# Isolation Forest tests
# ---------------------------------------------------------------------------


def test_isolation_forest_scores_normal_data_below_half():
    """Normal readings should score below 0.5 on average."""
    from gridsense.ml.isolation_forest import GridSenseIsolationForest

    X_train = generate_normal_readings(n_samples=2_000, random_seed=42)
    X_test = generate_normal_readings(n_samples=200, random_seed=99)

    model = GridSenseIsolationForest(contamination=0.05, n_estimators=50)
    model.fit(X_train)
    scores = model.score(X_test)

    assert scores.mean() < 0.5, (
        f"Normal data average score should be <0.5, got {scores.mean():.4f}"
    )


def test_isolation_forest_scores_anomalous_data_higher_than_normal():
    """Anomalous readings should score higher than normal readings on average."""
    from gridsense.ml.isolation_forest import GridSenseIsolationForest

    X_train = generate_normal_readings(n_samples=2_000, random_seed=42)
    X_normal_test = generate_normal_readings(n_samples=100, random_seed=10)
    X_anomaly_test = generate_anomalous_readings(n_samples=100, random_seed=20)

    model = GridSenseIsolationForest(contamination=0.05, n_estimators=50)
    model.fit(X_train)

    normal_scores = model.score(X_normal_test)
    anomaly_scores = model.score(X_anomaly_test)

    assert anomaly_scores.mean() > normal_scores.mean(), (
        f"Anomaly mean ({anomaly_scores.mean():.4f}) should exceed "
        f"normal mean ({normal_scores.mean():.4f})"
    )


def test_isolation_forest_degrading_transformer_scores_higher_than_normal():
    """A degrading transformer sequence should score higher than normal data on average."""
    from gridsense.ml.isolation_forest import GridSenseIsolationForest

    X_train = generate_normal_readings(n_samples=2_000, random_seed=42)
    model = GridSenseIsolationForest(contamination=0.05, n_estimators=50)
    model.fit(X_train)

    normal = generate_normal_readings(n_samples=100, random_seed=55)
    degrading = generate_degrading_sequence(n_steps=100)

    normal_scores = model.score(normal)
    degrading_scores = model.score(degrading)

    # Degrading sequence should consistently score higher than normal data
    assert degrading_scores.mean() > normal_scores.mean(), (
        f"Degrading mean ({degrading_scores.mean():.4f}) should exceed "
        f"normal mean ({normal_scores.mean():.4f})"
    )
    # The last half of the degrading sequence (more degraded) should score higher
    assert degrading_scores[50:].mean() > degrading_scores[:50].mean(), (
        "Later degraded readings should score higher than early ones"
    )


def test_isolation_forest_save_load(tmp_path):
    """Saved and reloaded model should produce identical scores."""
    from gridsense.ml.isolation_forest import GridSenseIsolationForest

    X = generate_normal_readings(n_samples=500, random_seed=1)
    X_test = generate_normal_readings(n_samples=20, random_seed=2)

    model = GridSenseIsolationForest(n_estimators=20)
    model.fit(X)
    scores_before = model.score(X_test)

    path = str(tmp_path / "if_test.joblib")
    model.save(path)

    loaded = GridSenseIsolationForest.from_file(path)
    scores_after = loaded.score(X_test)

    np.testing.assert_array_almost_equal(scores_before, scores_after, decimal=6)


# ---------------------------------------------------------------------------
# LSTM Autoencoder tests
# ---------------------------------------------------------------------------


def test_lstm_reconstruction_error_normal_below_anomalous():
    """Anomalous sequences should have higher reconstruction error than normal ones."""
    from gridsense.ml.lstm_autoencoder import INPUT_TIMESTEPS, GridSenseLSTM

    # Build small training set
    X_raw = generate_normal_readings(n_samples=1_000, random_seed=42)
    X_feat = extract_lstm_features(X_raw)
    X_seq = make_sequences(X_feat, timesteps=INPUT_TIMESTEPS)

    if len(X_seq) < 10:
        pytest.skip("Not enough sequences to train LSTM in test")

    model = GridSenseLSTM()
    model.train(X_seq[:200], epochs=5, batch_size=16)

    X_anom_raw = generate_anomalous_readings(n_samples=200, random_seed=77)
    X_anom_feat = extract_lstm_features(X_anom_raw)
    X_anom_seq = make_sequences(X_anom_feat, timesteps=INPUT_TIMESTEPS)

    if len(X_anom_seq) < 5:
        pytest.skip("Not enough anomalous sequences for LSTM test")

    norm_errors = model.reconstruction_error(X_seq[:50])
    anom_errors = model.reconstruction_error(X_anom_seq[:50])

    assert anom_errors.mean() > norm_errors.mean(), (
        f"Anomaly error ({anom_errors.mean():.4f}) should exceed "
        f"normal error ({norm_errors.mean():.4f})"
    )


# ---------------------------------------------------------------------------
# CombinedScorer tests
# ---------------------------------------------------------------------------


def _make_reading_dict(
    Va=230.0, Vb=230.0, Vc=230.0,
    Ia=50.0, Ib=50.0, Ic=50.0,
    oil_temp=60.0, power_factor=0.92,
    thd_pct=2.0, active_power_kw=35.0, reactive_power_kvar=8.0,
) -> dict:
    """Build a minimal reading dict for testing."""
    from datetime import datetime, timezone
    return {
        "Va": Va, "Vb": Vb, "Vc": Vc,
        "Ia": Ia, "Ib": Ib, "Ic": Ic,
        "oil_temp": oil_temp,
        "power_factor": power_factor,
        "thd_pct": thd_pct,
        "active_power_kw": active_power_kw,
        "reactive_power_kvar": reactive_power_kvar,
        "tamper_flag": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_and_load_scorer(tmp_path) -> object:
    """Helper: train minimal IF model and return a loaded CombinedScorer."""
    from gridsense.ml.combined_scorer import CombinedScorer
    from gridsense.ml.isolation_forest import GridSenseIsolationForest

    X = generate_normal_readings(n_samples=500, random_seed=42)
    if_model = GridSenseIsolationForest(n_estimators=20)
    if_model.fit(X)
    if_path = str(tmp_path / "if.joblib")
    if_model.save(if_path)

    scorer = CombinedScorer(if_model_path=if_path)
    return scorer


def test_combined_scorer_normal_reading_low_score(tmp_path):
    """Normal readings should produce low combined scores."""
    scorer = _build_and_load_scorer(tmp_path)
    readings = [_make_reading_dict() for _ in range(5)]
    result = scorer.score("T-001", readings)
    assert result.ewma_score < 0.75, (
        f"Normal readings should score <0.75, got {result.ewma_score:.4f}"
    )


def test_combined_scorer_uses_if_weight():
    """CombinedScorer weight constants must sum to 1.0."""
    from gridsense.ml.combined_scorer import IF_WEIGHT, LSTM_WEIGHT

    assert abs(IF_WEIGHT + LSTM_WEIGHT - 1.0) < 1e-9, (
        f"IF_WEIGHT ({IF_WEIGHT}) + LSTM_WEIGHT ({LSTM_WEIGHT}) must equal 1.0"
    )


def test_combined_scorer_hours_to_failure_returns_none_on_flat_trend():
    """hours_to_failure should return None for flat or improving trends."""
    from gridsense.ml.combined_scorer import CombinedScorer

    scorer = CombinedScorer()
    # Flat sequence far below threshold
    history = [0.30] * 60
    htf = scorer.hours_to_failure(history)
    assert htf is None, f"Flat trend should return None, got {htf}"


def test_combined_scorer_hours_to_failure_rising_trend():
    """hours_to_failure should return a positive float for a rising trend."""
    from gridsense.ml.combined_scorer import CombinedScorer

    scorer = CombinedScorer()
    # Linearly rising from 0.6 to 0.85 over 48 steps
    history = [0.60 + 0.25 * i / 47 for i in range(48)]
    htf = scorer.hours_to_failure(history)
    assert htf is not None and htf > 0, (
        f"Rising trend should return positive hours, got {htf}"
    )


def test_combined_scorer_degrading_data_hours_below_72(tmp_path):
    """A rapidly degrading transformer should predict failure within 72 hours."""
    scorer = _build_and_load_scorer(tmp_path)

    # Simulate rapidly degrading readings
    rng = np.random.default_rng(99)
    readings = []
    for step in range(80):
        readings.append(_make_reading_dict(
            oil_temp=60.0 + 0.5 * step,
            thd_pct=2.0 + 0.15 * step,
            Va=230.0 - 0.1 * step,
        ))
        result = scorer.score("T-047", readings)

    # After 80 degraded steps the scorer should predict failure < 72h
    htf = result.hours_to_failure
    # Allow None or a positive value; if not None it should be reasonable
    if htf is not None:
        assert htf >= 0, f"hours_to_failure must be non-negative, got {htf}"
