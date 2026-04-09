"""System health and dashboard summary endpoints."""
from __future__ import annotations

import time

from fastapi import APIRouter

from gridsense.db import repository
from gridsense.db import store as _store

router = APIRouter(tags=["system"])
_start_time = time.time()


@router.get("/health")
async def health() -> dict:
    """Health check — returns status, uptime, and ingestion counters."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "transformers_tracked": len(_store.get_all_transformer_ids()),
        "total_readings": len(_store.STORE["transformer_readings"]),
        "total_alerts": len(_store.STORE["alerts"]),
        "total_work_orders": len(_store.STORE["work_orders"]),
    }


@router.get("/dashboard/summary")
async def dashboard_summary() -> dict:
    """Aggregate summary metrics for the Streamlit dashboard header cards."""
    return await repository.get_dashboard_summary()
