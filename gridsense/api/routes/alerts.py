"""Alert API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gridsense.db import repository
from gridsense.db import store as _store

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/active")
async def get_active_alerts() -> list:
    """Return all active (non-resolved) transformer alerts."""
    return await repository.get_active_alerts()


@router.get("/history")
async def get_alert_history() -> list:
    """Return the complete alert history."""
    return await repository.get_all_alerts()


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> dict:
    """Acknowledge an alert without resolving it."""
    with _store._lock:
        for alert in _store.STORE["alerts"]:
            if alert.get("id") == alert_id:
                alert["status"] = "ACKNOWLEDGED"
                return {"id": alert_id, "status": "ACKNOWLEDGED"}
    raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
