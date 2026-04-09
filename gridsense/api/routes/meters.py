"""Meter and NTL detection API endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from gridsense.db import repository

router = APIRouter(tags=["meters"])


@router.get("/meters/{transformer_id}")
async def get_meter_readings(transformer_id: str) -> list:
    """Get last 96 meter readings for all smart meters on a transformer."""
    return await repository.get_meter_readings(transformer_id, n=96)


@router.get("/theft/detected")
async def get_theft_detections() -> list:
    """Return meters flagged for Non-Technical Loss (tamper_flag or >40% consumption drop)."""
    return await repository.get_theft_detections()
