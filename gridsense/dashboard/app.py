"""GridSense AI Streamlit dashboard."""
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
from gridsense.dashboard.components.metrics import score_to_colour, severity_badge, status_badge

API_BASE = "http://localhost:8000"
REFRESH_INTERVAL = 10

st.set_page_config(
    page_title="GridSense AI",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _client() -> httpx.Client:
    """Return a shared httpx client stored in session state."""
    if "http_client" not in st.session_state:
        st.session_state["http_client"] = httpx.Client(base_url=API_BASE, timeout=5.0)
    return st.session_state["http_client"]


def _get(path: str) -> Any:
    """Make a GET request and return JSON, or None on error."""
    try:
        resp = _client().get(path)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.warning(f"API call failed ({path}): {exc}")
        return None


def _post(path: str, body: dict) -> Any:
    """Make a POST request and return JSON, or None on error."""
    try:
        resp = _client().post(path, json=body)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"POST failed ({path}): {exc}")
        return None


def _put(path: str, body: dict) -> Any:
    """Make a PUT request and return JSON, or None on error."""
    try:
        resp = _client().put(path, json=body)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"PUT failed ({path}): {exc}")
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_drop_pct(reading: dict) -> str:
    """Render consumption drop in a user-friendly way."""
    drop = reading.get("consumption_drop_pct")
    if drop is None:
        return "Not available"
    return f"{_as_float(drop):.1f}%"


def _ntl_reason(reading: dict) -> str:
    """Explain why a meter was flagged in plain language."""
    drop = _as_float(reading.get("consumption_drop_pct"), 0.0)
    tamper = bool(reading.get("tamper_flag"))
    if tamper and drop > 40:
        return "The meter reported tampering and its power use suddenly dropped."
    if tamper:
        return "The meter itself raised a tamper signal."
    if drop > 40:
        return f"The meter's power use suddenly fell by {drop:.1f}%."
    return "This meter needs review because its latest reading looks abnormal."


def _ntl_next_step(reading: dict) -> str:
    """Return a simple recommended next step for operators."""
    if reading.get("tamper_flag"):
        return "Ask the field team to inspect the meter seal, wiring, and bypass condition."
    drop = _as_float(reading.get("consumption_drop_pct"), 0.0)
    if drop > 40:
        return "Check the site for sudden load loss, meter bypass, or communication issues."
    return "Review the latest readings and confirm whether site inspection is needed."


def _generate_operator_report(title: str, facts: str) -> str:
    """Generate a short operator report with the local LLM, or fallback text."""
    from gridsense.genai.local_llm import LocalLLMClient

    llm = LocalLLMClient()
    return llm.generate_operator_report(title=title, facts=facts)


def _show_operator_report_button(report_key: str, title: str, facts: str, button_label: str) -> None:
    """Render a button that generates and displays a short AI operator report."""
    report_store = st.session_state.setdefault("ai_reports", {})
    if st.button(button_label, key=f"btn_{report_key}"):
        report_store[report_key] = _generate_operator_report(title=title, facts=facts)
    report = report_store.get(report_key)
    if report:
        st.markdown("**AI Operator Report**")
        st.info(report)


st.sidebar.title("GridSense AI")
st.sidebar.markdown("Predictive Grid Fault Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Live Grid Overview",
        "Transformer Detail",
        "Alerts & Work Orders",
        "NTL Detection",
    ],
)

st.sidebar.divider()
st.sidebar.caption(f"API: {API_BASE}")
st.sidebar.caption(f"Auto-refresh: every {REFRESH_INTERVAL}s")


if page == "Live Grid Overview":
    st.title("Live Grid Overview")

    summary = _get("/dashboard/summary") or {}
    transformers = _get("/transformers") or []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transformers", summary.get("total_transformers", len(transformers)))
    c2.metric("Critical Alerts", summary.get("critical_alerts", 0))
    c3.metric("Warning Alerts", summary.get("warning_alerts", 0))
    c4.metric("Work Orders Pending", summary.get("work_orders_pending", 0))

    st.divider()

    if transformers:
        st.plotly_chart(make_transformer_grid(transformers), use_container_width=True)
    else:
        st.info("No transformer data yet. Start the simulator and the API ingestion flow.")

    st.divider()
    st.subheader("All Transformers")
    if transformers:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "ID": t["transformer_id"],
                    "Score": round(t.get("anomaly_score", 0.0), 4),
                    "Alert Level": t.get("alert_level", "NORMAL"),
                    "Oil Temp (C)": t.get("oil_temp") or "-",
                    "THD %": t.get("thd_pct") or "-",
                    "Hours to Failure": t.get("hours_to_failure") or "-",
                    "Last Updated": str(t.get("last_updated", ""))[:19],
                }
                for t in sorted(transformers, key=lambda item: item.get("anomaly_score", 0.0), reverse=True)
            ]
        )

        def _colour_row(row: Any) -> list[str]:
            colour = score_to_colour(float(row["Score"]))
            return [f"color: {colour}"] + [""] * (len(row) - 1)

        st.dataframe(df.style.apply(_colour_row, axis=1), use_container_width=True, height=400)
    else:
        st.info("Waiting for data...")

    time.sleep(REFRESH_INTERVAL)
    st.rerun()


elif page == "Transformer Detail":
    st.title("Transformer Detail")

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

    ewma = _as_float(score_data.get("ewma_score"), 0.0)
    htf = score_data.get("hours_to_failure")
    alert_level = score_data.get("alert_level", "NORMAL")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score", round(ewma, 4))
    m2.metric("Current Status", alert_level)
    m3.metric("Hours to Failure", f"{htf:.1f}h" if htf is not None else "N/A")
    m4.metric("Readings Available", len(readings))

    col_gauge, col_trend = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(make_score_gauge(ewma, selected), use_container_width=True)
    with col_trend:
        history = trend.get("score_history", [])
        if history:
            st.plotly_chart(make_score_trend(history, selected), use_container_width=True)
        else:
            st.info("Score history is still building.")

    st.divider()

    if readings:
        col_v, col_i = st.columns(2)
        with col_v:
            st.plotly_chart(make_voltage_chart(readings, selected), use_container_width=True)
        with col_i:
            st.plotly_chart(make_current_chart(readings, selected), use_container_width=True)

        st.subheader("Latest Reading")
        latest = readings[-1]
        import pandas as pd

        st.dataframe(pd.DataFrame([{k: v for k, v in latest.items() if k != "timestamp"}]), use_container_width=True)
    else:
        st.info("No readings available yet.")

    st.divider()
    st.subheader(f"Work Orders for {selected}")
    if work_orders:
        for wo in work_orders:
            wo_id = wo.get("id", "")
            with st.expander(f"{wo_id[:8]} - {wo.get('fault_type', '?')} [{wo.get('priority', '?')}]"):
                st.markdown(f"**Status:** {status_badge(wo.get('status', ''))}")
                st.markdown(f"**Severity:** {severity_badge(wo.get('severity', ''))}")
                st.markdown(f"**Evidence:** {wo.get('evidence', '')}")
                st.markdown(f"**Action:** {wo.get('recommended_action', '')}")
                st.markdown(f"**Tools:** {wo.get('tools_needed', '')}")
                st.markdown(f"**Estimated hours:** {wo.get('estimated_repair_hours', '')}")
                _show_operator_report_button(
                    report_key=f"transformer_wo_{wo_id}",
                    title=f"Transformer report for {wo.get('transformer_id', selected)}",
                    facts=(
                        f"Fault: {wo.get('fault_type', '')}. "
                        f"Severity: {wo.get('severity', '')}. "
                        f"Evidence: {wo.get('evidence', '')}. "
                        f"Action: {wo.get('recommended_action', '')}"
                    ),
                    button_label="Generate AI operator report",
                )
    else:
        st.info("No work orders for this transformer.")

    st.divider()
    if st.button(f"Generate AI Work Order for {selected}"):
        result = _post("/workorders", {"transformer_id": selected})
        if result:
            st.success(f"Work order {result.get('id', '')[:8]} created. AI diagnosis is running.")


elif page == "Alerts & Work Orders":
    st.title("Alerts & Work Orders")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Active Alerts")
        alerts = _get("/alerts/active") or []
        if alerts:
            for alert in alerts[:20]:
                badge = severity_badge(alert.get("severity", "NORMAL"))
                with st.expander(f"{badge} - {alert.get('transformer_id', '?')} (score: {alert.get('ewma_score', 0):.3f})"):
                    st.write(f"**ID:** {alert.get('id', '')[:12]}")
                    st.write(f"**Time:** {str(alert.get('timestamp', ''))[:19]}")
                    st.write(f"**Score:** {alert.get('anomaly_score', 0):.4f}")
                    st.write(f"**Risk score:** {alert.get('ewma_score', 0):.4f}")
                    htf = alert.get("hours_to_failure")
                    st.write(f"**Hours to failure:** {f'{htf:.1f}h' if htf else 'N/A'}")
                    st.write(f"**Message:** {alert.get('message', '')}")
                    st.write(f"**Status:** {alert.get('status', '')}")

                    if st.button("Create Work Order", key=f"wo_{alert.get('id', '')}"):
                        result = _post(
                            "/workorders",
                            {
                                "transformer_id": alert.get("transformer_id"),
                                "alert_id": alert.get("id"),
                            },
                        )
                        if result:
                            st.success("Work order created.")
        else:
            st.success("No active alerts. All transformers look healthy.")

    with col_right:
        st.subheader("Work Orders")
        work_orders = _get("/workorders") or []
        if work_orders:
            for wo in work_orders[:20]:
                wo_id = wo.get("id", "")
                with st.expander(f"{status_badge(wo.get('status', ''))} - {wo.get('transformer_id', '?')} [{wo.get('priority', '?')}]"):
                    st.write(f"**ID:** {wo_id[:12]}")
                    st.write(f"**Fault:** {wo.get('fault_type', '')}")
                    st.write(f"**Severity:** {severity_badge(wo.get('severity', ''))}")
                    st.write(f"**Evidence:** {wo.get('evidence', '')}")
                    st.write(f"**Action:** {wo.get('recommended_action', '')}")
                    st.write(f"**Tools:** {wo.get('tools_needed', '')}")
                    st.write(f"**Estimated hours:** {wo.get('estimated_repair_hours', '')}")
                    st.write(f"**Created:** {str(wo.get('created_at', ''))[:19]}")
                    _show_operator_report_button(
                        report_key=f"work_order_{wo_id}",
                        title=f"Field report for {wo.get('transformer_id', '')}",
                        facts=(
                            f"Status: {wo.get('status', '')}. "
                            f"Fault: {wo.get('fault_type', '')}. "
                            f"Severity: {wo.get('severity', '')}. "
                            f"Evidence: {wo.get('evidence', '')}. "
                            f"Action: {wo.get('recommended_action', '')}"
                        ),
                        button_label="Generate AI field report",
                    )

                    c1, c2 = st.columns(2)
                    if c1.button("In Progress", key=f"prog_{wo_id}"):
                        _put(f"/workorders/{wo_id}/status", {"status": "IN_PROGRESS"})
                        st.rerun()
                    if c2.button("Resolve", key=f"resolve_{wo_id}"):
                        _put(f"/workorders/{wo_id}/status", {"status": "RESOLVED"})
                        st.rerun()
        else:
            st.info("No work orders yet.")


elif page == "NTL Detection":
    st.title("Non-Technical Loss (NTL) Detection")
    st.markdown(
        "This page highlights meters that may need inspection. "
        "A meter is flagged when it reports tampering or when its power use drops suddenly."
    )

    theft = _get("/theft/detected") or []

    if not theft:
        st.success("No suspicious meters found yet.")
        st.stop()

    import pandas as pd

    summary_rows = [
        {
            "Meter": item.get("meter_id", "?"),
            "Transformer": item.get("transformer_id", "?"),
            "Current Usage (kW)": round(_as_float(item.get("active_power_kw"), 0.0), 2),
            "Usage Drop": _format_drop_pct(item),
            "Why Flagged": _ntl_reason(item),
        }
        for item in theft
    ]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Meters Needing Inspection", len(theft))
        st.markdown("**What this means**")
        st.write("These meters should be checked by an operator or field team.")
        st.markdown("**Flagged meters**")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with c2:
        selected_meter = st.selectbox(
            "Review a flagged meter",
            options=[item.get("meter_id", "?") for item in theft],
        )
        selected_reading = next(
            (item for item in theft if item.get("meter_id") == selected_meter),
            theft[0],
        )
        transformer_id = selected_reading.get("transformer_id", "")
        meter_data = _get(f"/meters/{transformer_id}") or []

        top1, top2, top3 = st.columns(3)
        top1.metric("Meter", selected_reading.get("meter_id", "?"))
        top2.metric("Current Usage", f"{_as_float(selected_reading.get('active_power_kw'), 0.0):.2f} kW")
        top3.metric("Usage Drop", _format_drop_pct(selected_reading))

        st.markdown("**Why this meter was flagged**")
        st.info(_ntl_reason(selected_reading))

        st.markdown("**Recommended next step**")
        st.write(_ntl_next_step(selected_reading))

        st.markdown("**Compare this meter with nearby meters on the same transformer**")
        st.caption("Red bars are flagged meters. Green bars are normal meters. The dashed line shows typical current usage.")
        st.plotly_chart(make_consumption_heatmap(meter_data), use_container_width=True)

        report_facts = (
            f"Meter {selected_reading.get('meter_id', '?')} on transformer {transformer_id}. "
            f"Current usage {_as_float(selected_reading.get('active_power_kw'), 0.0):.2f} kW. "
            f"Usage drop {_format_drop_pct(selected_reading)}. "
            f"Tamper flag {'present' if selected_reading.get('tamper_flag') else 'not present'}. "
            f"Reason: {_ntl_reason(selected_reading)}. "
            f"Recommended next step: {_ntl_next_step(selected_reading)}"
        )
        _show_operator_report_button(
            report_key=f"ntl_{selected_reading.get('meter_id', '?')}",
            title=f"Inspection report for meter {selected_reading.get('meter_id', '?')}",
            facts=report_facts,
            button_label="Generate AI inspection report",
        )
