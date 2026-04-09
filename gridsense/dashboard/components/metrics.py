"""Helper functions for alert formatting and priority colours."""
from __future__ import annotations


def alert_colour(alert_level: str) -> str:
    """Return a CSS colour hex string for a given alert level."""
    return {"CRITICAL": "#dc2626", "WARNING": "#f59e0b", "NORMAL": "#16a34a"}.get(
        alert_level.upper(), "#6b7280"
    )


def priority_colour(priority: str) -> str:
    """Return a CSS colour hex string for a work-order priority."""
    return {"HIGH": "#dc2626", "MEDIUM": "#f59e0b", "LOW": "#16a34a"}.get(
        priority.upper(), "#6b7280"
    )


def status_badge(status: str) -> str:
    """Return an emoji badge for a work-order status."""
    return {"PENDING": "🟡 PENDING", "IN_PROGRESS": "🔵 IN PROGRESS", "RESOLVED": "✅ RESOLVED"}.get(
        status.upper(), status
    )


def severity_badge(severity: str) -> str:
    """Return a styled text badge for alert severity."""
    return {
        "CRITICAL": "🔴 CRITICAL",
        "WARNING": "🟠 WARNING",
        "NORMAL": "🟢 NORMAL",
    }.get(severity.upper(), severity)


def score_to_colour(score: float) -> str:
    """Map a 0-1 anomaly score to a traffic-light colour."""
    if score >= 0.90:
        return "#dc2626"  # red
    if score >= 0.75:
        return "#f59e0b"  # amber
    return "#16a34a"      # green
