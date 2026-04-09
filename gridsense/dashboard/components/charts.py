"""Plotly chart factory functions for the GridSense AI dashboard."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def make_score_gauge(score: float, transformer_id: str) -> go.Figure:
    """Build a Plotly gauge chart for the EWMA anomaly score.

    Args:
        score: EWMA anomaly score in [0.0, 1.0]
        transformer_id: label shown on the gauge
    Returns:
        Plotly Figure
    """
    colour = "#dc2626" if score >= 0.90 else "#f59e0b" if score >= 0.75 else "#16a34a"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score, 4),
            title={"text": f"Anomaly Score — {transformer_id}", "font": {"size": 14}},
            number={"font": {"color": colour, "size": 28}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": colour},
                "steps": [
                    {"range": [0, 0.75], "color": "#bbf7d0"},
                    {"range": [0.75, 0.90], "color": "#fef08a"},
                    {"range": [0.90, 1.0], "color": "#fecaca"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20))
    return fig


def make_score_trend(score_history: list[float], transformer_id: str) -> go.Figure:
    """Build a line chart of EWMA anomaly score over time.

    Args:
        score_history: list of EWMA scores (oldest first)
        transformer_id: chart title label
    Returns:
        Plotly Figure
    """
    x = list(range(len(score_history)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=score_history,
            mode="lines",
            name="EWMA Score",
            line={"color": "#3b82f6", "width": 2},
        )
    )
    fig.add_hline(y=0.75, line_dash="dash", line_color="#f59e0b", annotation_text="Warning 0.75")
    fig.add_hline(y=0.90, line_dash="dash", line_color="#dc2626", annotation_text="Critical 0.90")
    fig.update_layout(
        title=f"Anomaly Score Trend — {transformer_id}",
        xaxis_title="Timestep",
        yaxis_title="EWMA Score",
        yaxis={"range": [0, 1.05]},
        height=300,
        margin=dict(t=40, b=30, l=40, r=20),
        showlegend=False,
    )
    return fig


def make_voltage_chart(readings: list[dict], transformer_id: str) -> go.Figure:
    """Build a multi-line chart of Va/Vb/Vc voltages.

    Args:
        readings: list of reading dicts (most recent last)
        transformer_id: chart title label
    Returns:
        Plotly Figure
    """
    idx = list(range(len(readings)))
    fig = go.Figure()
    for phase, colour in [("Va", "#3b82f6"), ("Vb", "#f59e0b"), ("Vc", "#16a34a")]:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=[r.get(phase, 230.0) for r in readings],
                mode="lines",
                name=phase,
                line={"color": colour, "width": 1.5},
            )
        )
    fig.add_hline(y=216, line_dash="dot", line_color="#dc2626", annotation_text="Min 216V")
    fig.add_hline(y=244, line_dash="dot", line_color="#dc2626", annotation_text="Max 244V")
    fig.update_layout(
        title=f"Phase Voltages — {transformer_id}",
        xaxis_title="Reading",
        yaxis_title="Voltage (V)",
        height=280,
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


def make_current_chart(readings: list[dict], transformer_id: str) -> go.Figure:
    """Build a multi-line chart of Ia/Ib/Ic phase currents."""
    idx = list(range(len(readings)))
    fig = go.Figure()
    for phase, colour in [("Ia", "#3b82f6"), ("Ib", "#f59e0b"), ("Ic", "#16a34a")]:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=[r.get(phase, 50.0) for r in readings],
                mode="lines",
                name=phase,
                line={"color": colour, "width": 1.5},
            )
        )
    fig.update_layout(
        title=f"Phase Currents — {transformer_id}",
        xaxis_title="Reading",
        yaxis_title="Current (A)",
        height=280,
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


def make_transformer_grid(transformer_list: list[dict]) -> go.Figure:
    """Build a colour-coded grid of all 100 transformers (10×10 heatmap).

    Args:
        transformer_list: list of transformer dicts with transformer_id and anomaly_score
    Returns:
        Plotly Figure with a 10×10 heatmap tile grid
    """
    scores_by_id: dict[str, float] = {
        t["transformer_id"]: t.get("anomaly_score", 0.0) for t in transformer_list
    }

    grid_z: list[list[float]] = []
    grid_text: list[list[str]] = []
    for row in range(10):
        z_row: list[float] = []
        t_row: list[str] = []
        for col in range(10):
            n = row * 10 + col + 1
            tid = f"T-{n:03d}"
            s = scores_by_id.get(tid, 0.0)
            z_row.append(s)
            t_row.append(f"{tid}<br>{s:.3f}")
        grid_z.append(z_row)
        grid_text.append(t_row)

    fig = go.Figure(
        go.Heatmap(
            z=grid_z,
            text=grid_text,
            texttemplate="%{text}",
            textfont={"size": 8},
            colorscale=[
                [0.0, "#bbf7d0"],
                [0.75, "#fef08a"],
                [0.90, "#fca5a5"],
                [1.0, "#dc2626"],
            ],
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar={"title": "Score"},
        )
    )
    fig.update_layout(
        title="Transformer Health Grid (100 transformers)",
        height=500,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def make_consumption_heatmap(meter_readings: list[dict]) -> go.Figure:
    """Build a heatmap of meter consumption over time for NTL detection.

    Args:
        meter_readings: list of meter reading dicts
    Returns:
        Plotly Figure
    """
    if not meter_readings:
        fig = go.Figure()
        fig.update_layout(title="No meter data yet", height=250)
        return fig

    # Group by meter_id, take last 20 readings each
    from collections import defaultdict

    by_meter: dict[str, list[dict]] = defaultdict(list)
    for r in meter_readings:
        by_meter[r.get("meter_id", "?")].append(r)

    meter_ids = sorted(by_meter.keys())[:20]  # show top 20 meters
    max_pts = 20
    z: list[list[float]] = []
    for mid in meter_ids:
        readings_for_meter = by_meter[mid][-max_pts:]
        row = [r.get("active_power_kw", 0.0) for r in readings_for_meter]
        # Pad to max_pts
        row = [0.0] * (max_pts - len(row)) + row
        z.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            y=meter_ids,
            colorscale="RdYlGn",
            colorbar={"title": "kW"},
        )
    )
    fig.update_layout(
        title="Smart Meter Consumption Heatmap",
        xaxis_title="Timestep (recent →)",
        yaxis_title="Meter ID",
        height=max(250, len(meter_ids) * 22 + 80),
        margin=dict(t=40, b=40, l=80, r=20),
    )
    return fig
