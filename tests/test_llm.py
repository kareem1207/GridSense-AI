"""GenAI layer tests for GridSense AI."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from gridsense.genai.local_llm import LocalLLMClient
from gridsense.schemas.work_orders import WorkOrderPriority, WorkOrderStatus


# ---------------------------------------------------------------------------
# LocalLLMClient tests
# ---------------------------------------------------------------------------


def test_local_llm_handles_connection_refused_gracefully():
    """generate_diagnosis returns a non-empty string when the server is down."""
    client = LocalLLMClient(base_url="http://127.0.0.1:19999/v1")  # Nothing running here
    result = client.generate_diagnosis(
        system_prompt="You are a grid engineer.",
        user_prompt="transformer_id: T-047\newma_smoothed_score: 0.92",
    )
    assert isinstance(result, str)
    assert len(result) > 0, "generate_diagnosis must return a non-empty fallback string"


def test_local_llm_fallback_is_valid_json():
    """Fallback response when LLM is offline should be valid JSON."""
    client = LocalLLMClient(base_url="http://127.0.0.1:19999/v1")
    result = client.generate_diagnosis(
        system_prompt="You are a grid engineer.",
        user_prompt="transformer_id: T-047\newma_smoothed_score: 0.92",
    )
    data = json.loads(result)
    assert "fault_type" in data
    assert "severity" in data
    assert "priority" in data
    assert data["priority"] in ("HIGH", "MEDIUM", "LOW")


def test_local_llm_fallback_critical_score_gives_high_priority():
    """Fallback: ewma_score >= 0.90 should produce HIGH priority."""
    client = LocalLLMClient(base_url="http://127.0.0.1:19999/v1")
    result = client.generate_diagnosis(
        system_prompt="",
        user_prompt="transformer_id: T-001\newma_smoothed_score: 0.95",
    )
    data = json.loads(result)
    assert data["priority"] == "HIGH"
    assert data["severity"] == "CRITICAL"


def test_local_llm_fallback_warning_score_gives_medium_priority():
    """Fallback: ewma_score in [0.75, 0.90) should produce MEDIUM priority."""
    client = LocalLLMClient(base_url="http://127.0.0.1:19999/v1")
    result = client.generate_diagnosis(
        system_prompt="",
        user_prompt="transformer_id: T-023\newma_smoothed_score: 0.82",
    )
    data = json.loads(result)
    assert data["priority"] == "MEDIUM"
    assert data["severity"] == "WARNING"


def test_local_llm_is_available_returns_false_when_unreachable():
    """is_available() should return False when no server is running."""
    client = LocalLLMClient(base_url="http://127.0.0.1:19999/v1")
    assert client.is_available() is False


# ---------------------------------------------------------------------------
# DiagnosisAgent tests (always run — uses mocked LLM)
# ---------------------------------------------------------------------------


def _make_sample_readings(n: int = 5) -> list[dict]:
    """Build n minimal transformer reading dicts."""
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "transformer_id": "T-047",
            "timestamp": now,
            "Va": 215.0,  # voltage sag
            "Vb": 215.0,
            "Vc": 215.0,
            "Ia": 50.0,
            "Ib": 50.0,
            "Ic": 50.0,
            "oil_temp": 88.0,  # above critical
            "power_factor": 0.82,
            "thd_pct": 7.5,
            "active_power_kw": 35.0,
            "reactive_power_kvar": 8.0,
            "tamper_flag": False,
        }
        for _ in range(n)
    ]


def test_diagnosis_agent_returns_valid_work_order_schema_with_mocked_llm():
    """DiagnosisAgent returns a valid WorkOrderSchema even when LLM is offline."""
    from gridsense.genai.diagnosis_agent import DiagnosisAgent

    # Mock the KB to avoid ChromaDB startup
    mock_kb = MagicMock()
    mock_kb.retrieve_similar_faults.return_value = [
        "Transformer insulation breakdown at high oil temperature.",
        "THD above 5% indicates harmonic distortion.",
    ]

    # Use a LocalLLMClient pointed at a non-existent server → triggers fallback
    llm = LocalLLMClient(base_url="http://127.0.0.1:19999/v1")

    agent = DiagnosisAgent(kb=mock_kb, llm=llm)
    readings = _make_sample_readings(5)

    wo = agent.diagnose(
        transformer_id="T-047",
        anomaly_score=0.91,
        ewma_score=0.93,
        hours_to_failure=12.5,
        readings=readings,
        alert_id="test-alert-id",
    )

    # Validate schema fields
    assert wo.transformer_id == "T-047"
    assert wo.alert_id == "test-alert-id"
    assert isinstance(wo.fault_type, str) and len(wo.fault_type) > 0
    assert wo.severity in ("CRITICAL", "WARNING", "NORMAL")
    assert wo.priority in (WorkOrderPriority.HIGH, WorkOrderPriority.MEDIUM, WorkOrderPriority.LOW)
    assert wo.status == WorkOrderStatus.PENDING
    assert wo.estimated_repair_hours > 0


def test_diagnosis_agent_critical_score_gives_high_priority_work_order():
    """DiagnosisAgent with score >= 0.90 should produce HIGH priority work order."""
    from gridsense.genai.diagnosis_agent import DiagnosisAgent

    mock_kb = MagicMock()
    mock_kb.retrieve_similar_faults.return_value = ["Overheating fault guide."]

    llm = LocalLLMClient(base_url="http://127.0.0.1:19999/v1")
    agent = DiagnosisAgent(kb=mock_kb, llm=llm)
    readings = _make_sample_readings(5)

    wo = agent.diagnose(
        transformer_id="T-047",
        anomaly_score=0.92,
        ewma_score=0.94,
        hours_to_failure=8.0,
        readings=readings,
    )

    assert wo.priority == WorkOrderPriority.HIGH
    assert wo.severity == "CRITICAL"


def test_diagnosis_agent_parse_response_handles_json_in_markdown():
    """DiagnosisAgent._parse_response extracts JSON wrapped in markdown code fence."""
    from gridsense.genai.diagnosis_agent import DiagnosisAgent

    agent = DiagnosisAgent.__new__(DiagnosisAgent)
    response = (
        "Here is the diagnosis:\n"
        "```json\n"
        '{"fault_type": "Overheating", "severity": "CRITICAL", '
        '"evidence": "High temp", "recommended_action": "Shutdown", '
        '"tools_needed": "Camera", "estimated_repair_hours": 4.0, "priority": "HIGH"}\n'
        "```"
    )
    data = agent._parse_response(response)
    assert data["fault_type"] == "Overheating"
    assert data["priority"] == "HIGH"


def test_diagnosis_agent_parse_response_handles_bare_json():
    """DiagnosisAgent._parse_response handles plain JSON string."""
    from gridsense.genai.diagnosis_agent import DiagnosisAgent

    agent = DiagnosisAgent.__new__(DiagnosisAgent)
    payload = {
        "fault_type": "Phase Imbalance",
        "severity": "WARNING",
        "evidence": "Ia 78A vs Ib 42A",
        "recommended_action": "Load balancing",
        "tools_needed": "Clamp meter",
        "estimated_repair_hours": 3.0,
        "priority": "MEDIUM",
    }
    data = agent._parse_response(json.dumps(payload))
    assert data["fault_type"] == "Phase Imbalance"


def test_diagnosis_agent_parse_response_returns_fallback_on_garbage():
    """DiagnosisAgent._parse_response returns a safe fallback for unparseable input."""
    from gridsense.genai.diagnosis_agent import DiagnosisAgent

    agent = DiagnosisAgent.__new__(DiagnosisAgent)
    data = agent._parse_response("This is not JSON at all and has no braces")
    assert "fault_type" in data
    assert "priority" in data


# ---------------------------------------------------------------------------
# Knowledge base tests (no heavy model loads)
# ---------------------------------------------------------------------------


def test_knowledge_base_build_and_query(tmp_path):
    """KnowledgeBase builds from docs dir and returns relevant text chunks."""
    import shutil

    from gridsense.genai.knowledge_base import GridSenseKnowledgeBase

    # Copy real docs to a temp dir
    import os

    real_docs = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "gridsense",
        "genai",
        "knowledge_base",
        "docs",
    )
    if not os.path.exists(real_docs):
        pytest.skip("Knowledge base docs not found")

    kb = GridSenseKnowledgeBase(
        docs_dir=real_docs,
        persist_dir=str(tmp_path / "chroma"),
    )
    kb.build()

    results = kb.retrieve_similar_faults("transformer overheating oil temperature", top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("oil" in r.lower() or "temperature" in r.lower() for r in results)
