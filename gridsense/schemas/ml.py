"""Pydantic v2 schemas for ML scoring outputs."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class MLScoreSchema(BaseModel):
    """ML anomaly score result for a transformer."""

    transformer_id: str
    raw_score: float
    ewma_score: float
    alert_level: str
    hours_to_failure: Optional[float] = None
    score_history: list[float] = []


class PredictionSchema(BaseModel):
    """Failure prediction for a transformer."""

    transformer_id: str
    predicted_failure_in_hours: Optional[float]
    confidence_pct: float
    trend_direction: str  # "rising", "falling", "stable"
