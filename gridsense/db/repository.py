"""Async repository — wraps in-memory STORE for FastAPI async context."""
from __future__ import annotations
import asyncio
from gridsense.db import store


async def get_all_transformers() -> list[dict]:
    """Return all transformer IDs with their latest scores."""
    ids = await asyncio.to_thread(store.get_all_transformer_ids)
    result = []
    for tid in ids:
        score_data = await asyncio.to_thread(store.get_ml_score, tid)
        readings = await asyncio.to_thread(store.get_recent_transformer_readings, tid, 1)
        latest = readings[-1] if readings else {}
        result.append({
            "transformer_id": tid,
            "anomaly_score": score_data.get("ewma_score", 0.0) if score_data else 0.0,
            "alert_level": score_data.get("alert_level", "NORMAL") if score_data else "NORMAL",
            "oil_temp": latest.get("oil_temp"),
            "thd_pct": latest.get("thd_pct"),
            "hours_to_failure": score_data.get("hours_to_failure") if score_data else None,
            "last_updated": str(latest.get("timestamp", "")),
        })
    return result


async def get_transformer_readings(transformer_id: str, n: int = 48) -> list[dict]:
    """Return last n readings for a transformer."""
    return await asyncio.to_thread(store.get_recent_transformer_readings, transformer_id, n)


async def get_transformer_score(transformer_id: str) -> dict | None:
    """Return ML score dict for a transformer."""
    return await asyncio.to_thread(store.get_ml_score, transformer_id)


async def get_active_alerts() -> list[dict]:
    """Return active alerts."""
    return await asyncio.to_thread(store.get_active_alerts)


async def get_all_alerts() -> list[dict]:
    """Return all alerts."""
    return await asyncio.to_thread(store.get_all_alerts)


async def get_all_work_orders() -> list[dict]:
    """Return all work orders."""
    return await asyncio.to_thread(store.get_all_work_orders)


async def get_work_orders_for_transformer(transformer_id: str) -> list[dict]:
    """Return work orders for a specific transformer."""
    return await asyncio.to_thread(store.get_work_orders_for_transformer, transformer_id)


async def create_work_order(wo: dict) -> dict:
    """Add a work order to the store and return it."""
    await asyncio.to_thread(store.add_work_order, wo)
    return wo


async def update_work_order_status(work_order_id: str, status: str) -> bool:
    """Update work order status."""
    return await asyncio.to_thread(store.update_work_order_status, work_order_id, status)


async def get_meter_readings(transformer_id: str, n: int = 96) -> list[dict]:
    """Return meter readings for a transformer."""
    return await asyncio.to_thread(store.get_recent_meter_readings, transformer_id, n)


async def get_theft_detections() -> list[dict]:
    """Return theft-flagged meter readings."""
    return await asyncio.to_thread(store.get_theft_detections)


async def get_dashboard_summary() -> dict:
    """Return summary metrics for the dashboard."""
    all_alerts = await asyncio.to_thread(store.get_all_alerts)
    all_wos = await asyncio.to_thread(store.get_all_work_orders)
    all_ids = await asyncio.to_thread(store.get_all_transformer_ids)
    critical = sum(1 for a in all_alerts if a.get("severity") == "CRITICAL" and a.get("status") != "RESOLVED")
    warning = sum(1 for a in all_alerts if a.get("severity") == "WARNING" and a.get("status") != "RESOLVED")
    pending_wos = sum(1 for w in all_wos if w.get("status") == "PENDING")
    return {
        "total_transformers": len(all_ids),
        "critical_alerts": critical,
        "warning_alerts": warning,
        "work_orders_pending": pending_wos,
        "total_readings": len(store.STORE["transformer_readings"]),
    }
