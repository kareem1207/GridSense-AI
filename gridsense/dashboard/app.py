"""GridSense AI — Streamlit Dashboard.

4-page dashboard:
  1. Live Grid Overview   — health grid + top-risk table, auto-refreshes every 10s
  2. Transformer Detail   — drill-down charts, gauge, AI diagnosis
  3. Alerts & Work Orders — alert table + work order manager
  4. NTL Detection        — theft-flagged meters heatmap

Run with:
    streamlit run gridsense/dashboard/app.py
"""
from __future__ import annotations

import time
from typing import Any

import httpx
import streamlit as st

from gridsense.dashboard.components.charts import (
    make_consumption_heatmap,
    make_current_chart,
    make_score_gauge,
    make_score_trend,
    make_transformer_grid,
    make_voltage_chart,
)
from gridsense.dashboard.components.metrics import (
    alert_colour,
    priority_colour,
    score_to_colour,
    severity_badge,
    status_badge,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"
REFRESH_INTERVAL = 10  # seconds

st.set_page_config(
    page_title="GridSense AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# HTTP helpers (cached in session_state to avoid re-creating the client)
# ---------------------------------------------------------------------------

def _client() -> httpx.Client:
    """Return a shared httpx.Client stored in session state."""
    if "http_client" not in st.session_state:
        st.session_state["http_client"] = httpx.Client(
            base_url=API_BASE, timeout=5.0
        )
    return st.session_state["http_client"]


def _get(path: str) -> Any:
    """Make a GET request; return parsed JSON or an empty default on error."""
    try:
        resp = _client().get(path)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.warning(f"API call failed ({path}): {exc}")
        return None


def _post(path: str, body: dict) -> Any:
    """Make a POST request; return parsed JSON or None on error."""
    try:
        resp = _client().post(path, json=body)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"POST failed ({path}): {exc}")
        return None


def _put(path: str, body: dict) -> Any:
    """Make a PUT request; return parsed JSON or None on error."""
    try:
        resp = _client().put(path, json=body)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"PUT failed ({path}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("⚡ GridSense AI")
st.sidebar.markdown("Predictive Grid Fault Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🌐 Live Grid Overview",
        "🔍 Transformer Detail",
        "🚨 Alerts & Work Orders",
        "🕵️ NTL Detection",
    ],
)

st.sidebar.divider()
st.sidebar.caption(f"API: {API_BASE}")
st.sidebar.caption(f"Auto-refresh: every {REFRESH_INTERVAL}s")

# ---------------------------------------------------------------------------
# Page 1 — Live Grid Overview
# ---------------------------------------------------------------------------

if page == "🌐 Live Grid Overview":
    st.title("🌐 Live Grid Overview")

    summary = _get("/dashboard/summary") or {}
    transformers = _get("/transformers") or []

    # --- Metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transformers", summary.get("total_transformers", len(transformers)))
    c2.metric(
        "🔴 Critical Alerts",
        summary.get("critical_alerts", 0),
        delta=None,
    )
    c3.metric(
        "🟠 Warning Alerts",
        summary.get("warning_alerts", 0),
    )
    c4.metric(
        "🟡 Work Orders Pending",
        summary.get("work_orders_pending", 0),
    )

    st.divider()

    # --- Transformer health grid ---
    if transformers:
        st.plotly_chart(
            make_transformer_grid(transformers),
            use_container_width=True,
        )
    else:
        st.info("No transformer data yet — start the simulator and ingestion consumer.")

    st.divider()

    # --- Top-risk table ---
    st.subheader("All Transformers")
    if transformers:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "ID": t["transformer_id"],
                    "Score": round(t.get("anomaly_score", 0.0), 4),
                    "Alert Level": t.get("alert_level", "NORMAL"),
                    "Oil Temp (°C)": t.get("oil_temp") or "—",
                    "THD %": t.get("thd_pct") or "—",
                    "Hours to Failure": t.get("hours_to_failure") or "—",
                    "Last Updated": str(t.get("last_updated", ""))[:19],
                }
                for t in sorted(
                    transformers, key=lambda x: x.get("anomaly_score", 0.0), reverse=True
                )
            ]
        )

        def _colour_row(row: Any) -> list[str]:
            c = score_to_colour(float(row["Score"]))
            return [f"color: {c}"] + [""] * (len(row) - 1)

        st.dataframe(
            df.style.apply(_colour_row, axis=1),
            use_container_width=True,
            height=400,
        )
    else:
        st.info("Waiting for data…")

    # Auto-refresh
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


# ---------------------------------------------------------------------------
# Page 2 — Transformer Detail
# ---------------------------------------------------------------------------

elif page == "🔍 Transformer Detail":
    st.title("🔍 Transformer Detail")

    transformer_ids = [t["transformer_id"] for t in (_get("/transformers") or [])]
    if not transformer_ids:
        st.info("No transformer data yet. Start the simulator first.")
        st.stop()

    selected = st.selectbox("Select Transformer", transformer_ids)

    detail = _get(f"/transformers/{selected}") or {}
    trend = _get(f"/transformers/{selected}/trend") or {}
    readings = detail.get("readings", [])
    score_data = detail.get("score") or {}
    work_orders = detail.get("work_orders", [])

    ewma = score_data.get("ewma_score", 0.0)
    htf = score_data.get("hours_to_failure")
    alert_level = score_data.get("alert_level", "NORMAL")

    # Metric row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("EWMA Score", round(ewma, 4))
    m2.metric(
        "Alert Level",
        alert_level,
        delta=None,
    )
    m3.metric(
        "Hours to Failure",
        f"{htf:.1f}h" if htf is not None else "N/A",
        delta=None,
    )
    m4.metric("Readings in buffer", len(readings))

    # Gauge
    col_gauge, col_trend = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(make_score_gauge(ewma, selected), use_container_width=True)
    with col_trend:
        history = trend.get("score_history", [])
        if history:
            st.plotly_chart(make_score_trend(history, selected), use_container_width=True)
        else:
            st.info("Score history accumulating…")

    st.divider()

    # DTM charts
    if readings:
        col_v, col_i = st.columns(2)
        with col_v:
            st.plotly_chart(make_voltage_chart(readings, selected), use_container_width=True)
        with col_i:
            st.plotly_chart(make_current_chart(readings, selected), use_container_width=True)

        # Latest reading table
        st.subheader("Latest Reading")
        latest = readings[-1] if readings else {}
        if latest:
            import pandas as pd

            st.dataframe(
                pd.DataFrame([{k: v for k, v in latest.items() if k != "timestamp"}]),
                use_container_width=True,
            )
    else:
        st.info("No readings available yet.")

    st.divider()

    # Work orders for this transformer
    st.subheader(f"Work Orders for {selected}")
    if work_orders:
        for wo in work_orders:
            with st.expander(
                f"{wo.get('id','')[:8]}… — {wo.get('fault_type','?')} [{wo.get('priority','?')}]"
            ):
                st.markdown(f"**Status:** {status_badge(wo.get('status',''))}")
                st.markdown(f"**Severity:** {severity_badge(wo.get('severity',''))}")
                st.markdown(f"**Evidence:** {wo.get('evidence','')}")
                st.markdown(f"**Action:** {wo.get('recommended_action','')}")
                st.markdown(f"**Tools:** {wo.get('tools_needed','')}")
                st.markdown(f"**Est. hours:** {wo.get('estimated_repair_hours','')}")
    else:
        st.info("No work orders for this transformer.")

    # Create work order button
    st.divider()
    if st.button(f"⚙️ Generate AI Work Order for {selected}"):
        result = _post("/workorders", {"transformer_id": selected})
        if result:
            st.success(f"Work order {result.get('id','')[:8]}… created. AI diagnosis running…")


# ---------------------------------------------------------------------------
# Page 3 — Alerts & Work Orders
# ---------------------------------------------------------------------------

elif page == "🚨 Alerts & Work Orders":
    st.title("🚨 Alerts & Work Orders")

    col_left, col_right = st.columns(2)

    # --- Active Alerts ---
    with col_left:
        st.subheader("Active Alerts")
        alerts = _get("/alerts/active") or []
        if alerts:
            for alert in alerts[:20]:
                severity = alert.get("severity", "NORMAL")
                badge = severity_badge(severity)
                with st.expander(
                    f"{badge} — {alert.get('transformer_id','?')} "
                    f"(score: {alert.get('ewma_score', 0):.3f})"
                ):
                    st.write(f"**ID:** {alert.get('id','')[:12]}…")
                    st.write(f"**Time:** {str(alert.get('timestamp',''))[:19]}")
                    st.write(f"**Score:** {alert.get('anomaly_score',0):.4f}")
                    st.write(f"**EWMA:** {alert.get('ewma_score',0):.4f}")
                    htf = alert.get("hours_to_failure")
                    st.write(f"**Hours to failure:** {f'{htf:.1f}h' if htf else 'N/A'}")
                    st.write(f"**Message:** {alert.get('message','')}")
                    st.write(f"**Status:** {alert.get('status','')}")

                    if st.button(
                        f"Create Work Order",
                        key=f"wo_{alert.get('id','')}",
                    ):
                        r = _post(
                            "/workorders",
                            {
                                "transformer_id": alert.get("transformer_id"),
                                "alert_id": alert.get("id"),
                            },
                        )
                        if r:
                            st.success("Work order created!")
        else:
            st.success("No active alerts — all transformers healthy.")

    # --- Work Orders ---
    with col_right:
        st.subheader("Work Orders")
        wos = _get("/workorders") or []
        if wos:
            for wo in wos[:20]:
                with st.expander(
                    f"{status_badge(wo.get('status',''))} — "
                    f"{wo.get('transformer_id','?')} [{wo.get('priority','?')}]"
                ):
                    st.write(f"**ID:** {wo.get('id','')[:12]}…")
                    st.write(f"**Fault:** {wo.get('fault_type','')}")
                    st.write(f"**Severity:** {severity_badge(wo.get('severity',''))}")
                    st.write(f"**Evidence:** {wo.get('evidence','')}")
                    st.write(f"**Action:** {wo.get('recommended_action','')}")
                    st.write(f"**Tools:** {wo.get('tools_needed','')}")
                    st.write(f"**Est. hours:** {wo.get('estimated_repair_hours','')}")
                    st.write(f"**Created:** {str(wo.get('created_at',''))[:19]}")

                    wo_id = wo.get("id", "")
                    c1, c2 = st.columns(2)
                    if c1.button("▶ In Progress", key=f"prog_{wo_id}"):
                        _put(f"/workorders/{wo_id}/status", {"status": "IN_PROGRESS"})
                        st.rerun()
                    if c2.button("✅ Resolve", key=f"resolve_{wo_id}"):
                        _put(f"/workorders/{wo_id}/status", {"status": "RESOLVED"})
                        st.rerun()
        else:
            st.info("No work orders yet.")


# ---------------------------------------------------------------------------
# Page 4 — NTL Detection
# ---------------------------------------------------------------------------

elif page == "🕵️ NTL Detection":
    st.title("🕵️ Non-Technical Loss (NTL) Detection")

    st.markdown(
        "Monitors smart meter consumption for sudden drops, tamper flags, "
        "and direct-bypass signatures (e.g. Meter M-04702)."
    )

    theft = _get("/theft/detected") or []

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Flagged Meters", len(theft))
        if theft:
            st.subheader("Flagged Meters")
            for r in theft:
                drop = r.get("consumption_drop_pct") or 0.0
                tamper = r.get("tamper_flag", False)
                badge = "🔴" if tamper else "🟠"
                st.markdown(
                    f"{badge} **{r.get('meter_id','?')}** on {r.get('transformer_id','?')}"
                    f" — {r.get('active_power_kw', 0):.2f} kW"
                    f" (drop: {drop:.1f}%)"
                    f"{' 🚩TAMPER' if tamper else ''}"
                )
        else:
            st.success("No theft detected.")

    with col2:
        # Show consumption heatmap for the transformer with most theft flags
        if theft:
            transformer_id = theft[0].get("transformer_id", "T-047")
            meter_data = _get(f"/meters/{transformer_id}") or []
            st.subheader(f"Meter Consumption Heatmap — {transformer_id}")
            st.plotly_chart(
                make_consumption_heatmap(meter_data), use_container_width=True
            )
        else:
            st.info("No NTL data — run the simulator until timestep 50 to see theft detection.")
