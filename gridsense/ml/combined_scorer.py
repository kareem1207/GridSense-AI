"""Combined anomaly scorer with EWMA smoothing and failure time prediction."""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

IF_WEIGHT: float = 0.6
LSTM_WEIGHT: float = 0.4
EWMA_SPAN: int = 12
EWMA_ALPHA: float = 2.0 / (EWMA_SPAN + 1)  # ≈ 0.1538
WARNING_THRESHOLD: float = 0.75
CRITICAL_THRESHOLD: float = 0.90
MIN_READINGS_FOR_LSTM: int = 48
STEP_SECONDS: float = 5.0       # seconds between simulator readings
SCORE_HISTORY_LEN: int = 200    # max EWMA scores kept per transformer


class ScoredResult:
    """Result of one combined-scoring pass for a single transformer."""

    __slots__ = [
        "transformer_id",
        "raw_score",
        "ewma_score",
        "alert_level",
        "hours_to_failure",
        "if_score",
        "lstm_score",
    ]

    def __init__(
        self,
        transformer_id: str,
        raw_score: float,
        ewma_score: float,
        alert_level: str,
        hours_to_failure: Optional[float],
        if_score: float,
        lstm_score: float,
    ) -> None:
        """Initialise a ScoredResult."""
        self.transformer_id = transformer_id
        self.raw_score = raw_score
        self.ewma_score = ewma_score
        self.alert_level = alert_level
        self.hours_to_failure = hours_to_failure
        self.if_score = if_score
        self.lstm_score = lstm_score

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for STORE."""
        return {
            "transformer_id": self.transformer_id,
            "latest_score": self.raw_score,
            "ewma_score": self.ewma_score,
            "alert_level": self.alert_level,
            "hours_to_failure": self.hours_to_failure,
            "if_score": self.if_score,
            "lstm_score": self.lstm_score,
        }


class CombinedScorer:
    """Combines Isolation Forest + LSTM scores with EWMA smoothing.

    Score formula: combined = 0.6 × IF_score + 0.4 × LSTM_score
    EWMA: smoothed = α × raw + (1 − α) × prev  (α = 2 / 13 ≈ 0.154, span=12)
    Failure prediction: LinearRegression on last 48 EWMA scores projected to 0.90.
    """

    def __init__(
        self,
        if_model_path: Optional[str] = None,
        lstm_model_path: Optional[str] = None,
    ) -> None:
        """Initialise scorer; optionally load models from disk immediately.

        Args:
            if_model_path: path to isolation_forest.joblib
            lstm_model_path: base path to lstm_autoencoder (without extension)
        """
        self._if = None
        self._lstm = None
        self._ewma_state: dict[str, float] = {}
        self._score_history: dict[str, deque] = {}
        self._is_ready: bool = False

        if if_model_path:
            self._load_models(if_model_path, lstm_model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self, if_path: str, lstm_path: Optional[str]) -> None:
        """Load IF (required) and LSTM (optional) from disk."""
        from gridsense.ml.isolation_forest import GridSenseIsolationForest

        self._if = GridSenseIsolationForest.from_file(if_path)
        logger.info("IsolationForest loaded for CombinedScorer")

        if lstm_path:
            try:
                from gridsense.ml.lstm_autoencoder import GridSenseLSTM

                self._lstm = GridSenseLSTM.from_file(lstm_path)
                logger.info("LSTM loaded for CombinedScorer")
            except Exception as exc:
                logger.warning("LSTM load failed (%s). Running IF-only scoring.", exc)

        self._is_ready = True

    def load_models(self, if_path: str, lstm_path: Optional[str] = None) -> None:
        """Explicitly load models (call this if paths not passed to __init__)."""
        self._load_models(if_path, lstm_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if not self._is_ready:
            raise RuntimeError("CombinedScorer models not loaded. Call load_models() first.")

    def _get_if_features(self, readings: list[dict]) -> np.ndarray:
        """Extract 11-feature row from the latest reading for IF scoring."""
        r = readings[-1]
        return np.array(
            [[
                r.get("Va", 230.0),
                r.get("Vb", 230.0),
                r.get("Vc", 230.0),
                r.get("Ia", 50.0),
                r.get("Ib", 50.0),
                r.get("Ic", 50.0),
                r.get("oil_temp", 60.0),
                r.get("power_factor", 0.92),
                r.get("thd_pct", 2.0),
                r.get("active_power_kw", 35.0),
                r.get("reactive_power_kvar", 8.0),
            ]],
            dtype=np.float64,
        )

    def _get_lstm_sequence(self, readings: list[dict]) -> np.ndarray:
        """Build a (1, 48, 7) sequence from last 48 readings for LSTM scoring.

        Features: [Va, Ib, oil_temp, power_factor, thd_pct, active_power_kw, reactive_power_kvar]
        """
        seq = readings[-MIN_READINGS_FOR_LSTM:]
        arr = np.array(
            [[
                r.get("Va", 230.0),
                r.get("Ib", 50.0),
                r.get("oil_temp", 60.0),
                r.get("power_factor", 0.92),
                r.get("thd_pct", 2.0),
                r.get("active_power_kw", 35.0),
                r.get("reactive_power_kvar", 8.0),
            ] for r in seq],
            dtype=np.float64,
        )
        return arr.reshape(1, MIN_READINGS_FOR_LSTM, 7)

    def _compute_ewma(self, transformer_id: str, raw_score: float) -> float:
        """Update and return EWMA-smoothed score for a transformer."""
        prev = self._ewma_state.get(transformer_id, raw_score)
        ewma = EWMA_ALPHA * raw_score + (1.0 - EWMA_ALPHA) * prev
        self._ewma_state[transformer_id] = ewma
        return ewma

    def _update_history(self, transformer_id: str, ewma_score: float) -> list[float]:
        """Append ewma_score to the per-transformer history deque; return as list."""
        if transformer_id not in self._score_history:
            self._score_history[transformer_id] = deque(maxlen=SCORE_HISTORY_LEN)
        self._score_history[transformer_id].append(ewma_score)
        return list(self._score_history[transformer_id])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hours_to_failure(self, score_history: list[float]) -> Optional[float]:
        """Predict hours until the EWMA score reaches CRITICAL_THRESHOLD (0.90).

        Fits LinearRegression on the last 48 EWMA values.
        Returns None when the trend is flat/improving or fewer than 48 points exist.

        Args:
            score_history: list of historical EWMA scores (oldest first)
        Returns:
            Predicted hours as a float, 0.0 if already past threshold, or None.
        """
        if len(score_history) < 48:
            return None

        y = np.array(score_history[-48:], dtype=np.float64)
        x = np.arange(len(y), dtype=np.float64).reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        slope = float(reg.coef_[0])

        if slope <= 0:
            return None  # Not degrading

        intercept = float(reg.intercept_)
        steps_to_threshold = (CRITICAL_THRESHOLD - intercept) / slope
        steps_from_now = steps_to_threshold - (len(y) - 1)

        if steps_from_now <= 0:
            return 0.0  # Already at or past the threshold

        hours = (steps_from_now * STEP_SECONDS) / 3600.0
        return round(hours, 2)

    def score(self, transformer_id: str, readings: list[dict]) -> ScoredResult:
        """Compute a combined anomaly score for one transformer.

        Steps:
          1. IF score on latest single reading.
          2. LSTM score on last 48-reading sequence (skipped if <48 available).
          3. Weighted combination: 0.6×IF + 0.4×LSTM (IF-only when LSTM unavailable).
          4. EWMA smoothing (span=12).
          5. Alert-level classification.
          6. Hours-to-failure projection (LinearRegression on last 48 EWMA values).

        Args:
            transformer_id: e.g. "T-047"
            readings: list of recent reading dicts, most-recent last
        Returns:
            ScoredResult populated with all components
        """
        self._ensure_ready()

        if not readings:
            return ScoredResult(transformer_id, 0.0, 0.0, "NORMAL", None, 0.0, 0.0)

        # Step 1: IF
        if_features = self._get_if_features(readings)
        if_score = float(self._if.score(if_features)[0])

        # Step 2: LSTM (conditional)
        lstm_score = 0.0
        use_lstm = self._lstm is not None and len(readings) >= MIN_READINGS_FOR_LSTM
        if use_lstm:
            seq = self._get_lstm_sequence(readings)
            lstm_score = float(self._lstm.reconstruction_error(seq)[0])

        # Step 3: Combined
        raw_score = (
            IF_WEIGHT * if_score + LSTM_WEIGHT * lstm_score
            if use_lstm
            else if_score
        )

        # Step 4: EWMA
        ewma = self._compute_ewma(transformer_id, raw_score)

        # Step 5: Alert level
        if ewma >= CRITICAL_THRESHOLD:
            alert_level = "CRITICAL"
        elif ewma >= WARNING_THRESHOLD:
            alert_level = "WARNING"
        else:
            alert_level = "NORMAL"

        # Step 6: History + prediction
        history = self._update_history(transformer_id, ewma)
        htf = self.hours_to_failure(history)

        return ScoredResult(transformer_id, raw_score, ewma, alert_level, htf, if_score, lstm_score)
