"""In-memory data store for GridSense AI — shared across all layers."""
from __future__ import annotations
import threading
from collections import deque
from typing import Any

_lock = threading.Lock()

STORE: dict[str, Any] = {
    "transformer_readings": deque(maxlen=10_000),
    "meter_readings": deque(maxlen=200_000),
    "alerts": deque(maxlen=1_000),
    "work_orders": deque(maxlen=500),
    "ml_scores": {},
    "_seen_transformers": set(),
    "_seen_meters": set(),
}


def append_transformer_reading(record: dict) -> None:
    """Append a transformer reading dict to the store."""
    with _lock:
        STORE["transformer_readings"].append(record)
        if "transformer_id" in record:
            STORE["_seen_transformers"].add(record["transformer_id"])


def append_meter_reading(record: dict) -> None:
    """Append a meter reading dict to the store."""
    with _lock:
        STORE["meter_readings"].append(record)
        if "meter_id" in record:
            STORE["_seen_meters"].add(record["meter_id"])
        if "transformer_id" in record:
            STORE["_seen_transformers"].add(record["transformer_id"])


def get_recent_transformer_readings(transformer_id: str, n: int = 48) -> list[dict]:
    """Return last n readings for a transformer, most recent last."""
    with _lock:
        readings = [r for r in STORE["transformer_readings"] if r.get("transformer_id") == transformer_id]
    return readings[-n:]


def get_recent_meter_readings(transformer_id: str, n: int = 96) -> list[dict]:
    """Return last n meter readings for meters attached to a transformer."""
    with _lock:
        readings = [r for r in STORE["meter_readings"] if r.get("transformer_id") == transformer_id]
    return readings[-n:]


def get_all_transformer_ids() -> list[str]:
    """Return sorted list of all seen transformer IDs."""
    with _lock:
        return sorted(STORE["_seen_transformers"])


def add_alert(alert: dict) -> None:
    """Add an alert to the store (prepend = most recent first)."""
    with _lock:
        STORE["alerts"].appendleft(alert)


def add_work_order(wo: dict) -> None:
    """Add a work order to the store (prepend = most recent first)."""
    with _lock:
        STORE["work_orders"].appendleft(wo)


def get_active_alerts() -> list[dict]:
    """Return all non-resolved alerts."""
    with _lock:
        return [a for a in STORE["alerts"] if a.get("status") != "RESOLVED"]


def get_all_alerts() -> list[dict]:
    """Return all alerts as a list."""
    with _lock:
        return list(STORE["alerts"])


def get_all_work_orders() -> list[dict]:
    """Return all work orders as a list."""
    with _lock:
        return list(STORE["work_orders"])


def get_work_orders_for_transformer(transformer_id: str) -> list[dict]:
    """Return all work orders for a specific transformer."""
    with _lock:
        return [w for w in STORE["work_orders"] if w.get("transformer_id") == transformer_id]


def update_ml_score(transformer_id: str, score_data: dict) -> None:
    """Update ML score data for a transformer."""
    with _lock:
        STORE["ml_scores"][transformer_id] = score_data
        STORE["_seen_transformers"].add(transformer_id)


def get_ml_score(transformer_id: str) -> dict | None:
    """Return ML score dict for a transformer, or None if not scored yet."""
    with _lock:
        return STORE["ml_scores"].get(transformer_id)


def update_work_order_status(work_order_id: str, status: str) -> bool:
    """Update a work order status. Returns True if found, False otherwise."""
    with _lock:
        for wo in STORE["work_orders"]:
            if wo.get("id") == work_order_id:
                wo["status"] = status
                return True
    return False


def get_all_meter_ids() -> list[str]:
    """Return sorted list of all seen meter IDs."""
    with _lock:
        return sorted(STORE["_seen_meters"])


def get_theft_detections() -> list[dict]:
    """Return the latest flagged reading per meter, newest first."""
    with _lock:
        result = []
        seen_meters: set[str] = set()
        for r in reversed(STORE["meter_readings"]):
            mid = r.get("meter_id", "")
            if mid in seen_meters:
                continue
            if r.get("tamper_flag") or (r.get("consumption_drop_pct") or 0) > 40:
                result.append(r)
                seen_meters.add(mid)
    return result
