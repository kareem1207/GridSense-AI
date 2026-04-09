"""FastAPI backend tests for GridSense AI."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from gridsense.api.main import app
from gridsense.db import store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_store():
    """Reset STORE to empty state before each test."""
    from collections import deque

    with store._lock:
        store.STORE["transformer_readings"] = deque(maxlen=10_000)
        store.STORE["meter_readings"] = deque(maxlen=200_000)
        store.STORE["alerts"] = deque(maxlen=1_000)
        store.STORE["work_orders"] = deque(maxlen=500)
        store.STORE["ml_scores"] = {}
        store.STORE["_seen_transformers"] = set()
        store.STORE["_seen_meters"] = set()
    yield


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client (no lifespan events fired)."""
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def populated_client() -> TestClient:
    """Test client with pre-seeded STORE data."""
    now = datetime.now(timezone.utc).isoformat()

    # Seed transformer readings
    for tid_n in range(1, 6):
        tid = f"T-{tid_n:03d}"
        for _ in range(10):
            store.append_transformer_reading({
                "transformer_id": tid,
                "timestamp": now,
                "Va": 230.0, "Vb": 230.0, "Vc": 230.0,
                "Ia": 50.0, "Ib": 50.0, "Ic": 50.0,
                "oil_temp": 60.0, "power_factor": 0.92,
                "thd_pct": 2.0, "active_power_kw": 35.0,
                "reactive_power_kvar": 8.0, "tamper_flag": False,
            })

    # Seed an alert
    store.add_alert({
        "id": str(uuid.uuid4()),
        "transformer_id": "T-001",
        "timestamp": now,
        "severity": "WARNING",
        "anomaly_score": 0.80,
        "ewma_score": 0.80,
        "hours_to_failure": 36.0,
        "status": "ACTIVE",
        "message": "Test alert",
    })

    # Seed a work order
    store.add_work_order({
        "id": str(uuid.uuid4()),
        "transformer_id": "T-001",
        "alert_id": None,
        "created_at": now,
        "status": "PENDING",
        "fault_type": "Test Fault",
        "severity": "WARNING",
        "evidence": "Test evidence",
        "recommended_action": "Test action",
        "tools_needed": "Multimeter",
        "estimated_repair_hours": 4.0,
        "priority": "MEDIUM",
    })

    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health_returns_200_and_ok(client):
    """/health must return HTTP 200 with status='ok'."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------


def test_list_transformers_empty_store_returns_list(client):
    """/transformers returns an empty list when no data has been ingested."""
    resp = client.get("/transformers")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_list_transformers_with_data(populated_client):
    """/transformers returns one entry per seen transformer."""
    resp = populated_client.get("/transformers")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5
    ids = {t["transformer_id"] for t in data}
    assert "T-001" in ids


def test_get_transformer_404_for_unknown(populated_client):
    """/transformers/{id} returns 404 for an unknown transformer."""
    resp = populated_client.get("/transformers/T-999")
    assert resp.status_code == 404


def test_get_transformer_detail(populated_client):
    """/transformers/T-001 returns readings, score, and work_orders keys."""
    resp = populated_client.get("/transformers/T-001")
    assert resp.status_code == 200
    data = resp.json()
    assert "readings" in data
    assert "score" in data
    assert "work_orders" in data


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


def test_alerts_active_returns_list(client):
    """/alerts/active returns a list (empty is fine)."""
    resp = client.get("/alerts/active")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_alerts_active_contains_seeded_alert(populated_client):
    """/alerts/active returns the seeded WARNING alert."""
    resp = populated_client.get("/alerts/active")
    assert resp.status_code == 200
    alerts = resp.json()
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "WARNING"
    assert alerts[0]["transformer_id"] == "T-001"


def test_alerts_history_returns_list(populated_client):
    """/alerts/history returns full history list."""
    resp = populated_client.get("/alerts/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1


# ---------------------------------------------------------------------------
# Work Orders
# ---------------------------------------------------------------------------


def test_list_work_orders_returns_list(client):
    """/workorders returns a list."""
    resp = client.get("/workorders")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_create_work_order_requires_transformer_id(populated_client):
    """POST /workorders without transformer_id returns 400."""
    resp = populated_client.post("/workorders", json={})
    assert resp.status_code == 400


def test_create_work_order_unknown_transformer_returns_404(populated_client):
    """POST /workorders with an unknown transformer returns 404."""
    resp = populated_client.post("/workorders", json={"transformer_id": "T-999"})
    assert resp.status_code == 404


def test_create_work_order_valid_transformer(populated_client):
    """POST /workorders with a valid transformer returns 200 and a work order dict."""
    resp = populated_client.post("/workorders", json={"transformer_id": "T-001"})
    assert resp.status_code == 200
    wo = resp.json()
    assert wo["transformer_id"] == "T-001"
    assert "id" in wo
    assert wo["status"] == "PENDING"


def test_update_work_order_status(populated_client):
    """PUT /workorders/{id}/status updates a work order status."""
    # Get the seeded work order id
    wos = populated_client.get("/workorders").json()
    assert wos, "Expected at least one work order from fixture"
    wo_id = wos[0]["id"]

    resp = populated_client.put(f"/workorders/{wo_id}/status", json={"status": "IN_PROGRESS"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "IN_PROGRESS"


def test_update_work_order_invalid_status(populated_client):
    """PUT /workorders/{id}/status with invalid status returns 400."""
    wos = populated_client.get("/workorders").json()
    wo_id = wos[0]["id"]
    resp = populated_client.put(f"/workorders/{wo_id}/status", json={"status": "INVALID"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Dashboard summary
# ---------------------------------------------------------------------------


def test_dashboard_summary_has_expected_keys(populated_client):
    """/dashboard/summary returns all required metric keys."""
    resp = populated_client.get("/dashboard/summary")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("total_transformers", "critical_alerts", "warning_alerts", "work_orders_pending"):
        assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Theft detection
# ---------------------------------------------------------------------------


def test_theft_detected_returns_list(client):
    """/theft/detected returns a list."""
    resp = client.get("/theft/detected")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_theft_detected_flags_tamper_meter():
    """/theft/detected includes meters with tamper_flag=True."""
    now = datetime.now(timezone.utc).isoformat()
    store.append_meter_reading({
        "meter_id": "M-04702",
        "transformer_id": "T-047",
        "timestamp": now,
        "active_power_kw": 0.3,
        "reactive_power_kvar": 0.1,
        "tamper_flag": True,
        "consumption_drop_pct": 91.0,
    })

    client = TestClient(app)
    resp = client.get("/theft/detected")
    assert resp.status_code == 200
    flagged = resp.json()
    meter_ids = [m["meter_id"] for m in flagged]
    assert "M-04702" in meter_ids
