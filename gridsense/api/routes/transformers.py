"""Transformer API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gridsense.db import repository
from gridsense.db import store as _store

router = APIRouter(prefix="/transformers", tags=["transformers"])


@router.get("")
async def list_transformers() -> list:
    """List all transformers with latest anomaly scores and key metrics."""
    return await repository.get_all_transformers()


@router.get("/{transformer_id}")
async def get_transformer(transformer_id: str) -> dict:
    """Get transformer detail: last 48 readings, ML score, and work orders."""
    ids = _store.get_all_transformer_ids()
    if ids and transformer_id not in ids:
        raise HTTPException(status_code=404, detail=f"Transformer {transformer_id} not found")
    readings = await repository.get_transformer_readings(transformer_id, n=48)
    score = await repository.get_transformer_score(transformer_id)
    work_orders = await repository.get_work_orders_for_transformer(transformer_id)
    return {
        "transformer_id": transformer_id,
        "readings": readings,
        "score": score,
        "work_orders": work_orders,
    }


@router.get("/{transformer_id}/trend")
async def get_transformer_trend(transformer_id: str) -> dict:
    """Get anomaly score trend history suitable for charting."""
    score_data = _store.get_ml_score(transformer_id) or {}
    return {
        "transformer_id": transformer_id,
        "score_history": score_data.get("score_history", []),
        "ewma_score": score_data.get("ewma_score", 0.0),
        "hours_to_failure": score_data.get("hours_to_failure"),
        "alert_level": score_data.get("alert_level", "NORMAL"),
    }
