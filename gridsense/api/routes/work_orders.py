"""Work order API endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from gridsense.db import repository
from gridsense.db import store as _store
from gridsense.schemas.work_orders import WorkOrderStatus

router = APIRouter(prefix="/workorders", tags=["work_orders"])


@router.get("")
async def list_work_orders() -> list:
    """List all work orders."""
    return await repository.get_all_work_orders()


@router.post("")
async def create_work_order(body: dict, background_tasks: BackgroundTasks) -> dict:
    """Create a placeholder work order and trigger AI diagnosis in the background."""
    transformer_id = body.get("transformer_id")
    if not transformer_id:
        raise HTTPException(status_code=400, detail="transformer_id is required")

    ids = _store.get_all_transformer_ids()
    if ids and transformer_id not in ids:
        raise HTTPException(status_code=404, detail=f"Transformer {transformer_id} not found")

    wo: dict = {
        "id": str(uuid.uuid4()),
        "transformer_id": transformer_id,
        "alert_id": body.get("alert_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": WorkOrderStatus.PENDING.value,
        "fault_type": "Pending AI Diagnosis",
        "severity": "WARNING",
        "evidence": "Work order created via API. AI diagnosis in progress.",
        "recommended_action": "Awaiting AI diagnosis. Stand by for updated instructions.",
        "tools_needed": "TBD — awaiting AI diagnosis",
        "estimated_repair_hours": 4.0,
        "priority": "MEDIUM",
    }
    _store.add_work_order(wo)
    background_tasks.add_task(_run_background_diagnosis, transformer_id, wo["id"])
    return wo


async def _run_background_diagnosis(transformer_id: str, work_order_id: str) -> None:
    """Background task: run AI diagnosis and update the placeholder work order."""
    try:
        from gridsense.genai.diagnosis_agent import DiagnosisAgent
        from gridsense.genai.knowledge_base import GridSenseKnowledgeBase
        from gridsense.genai.local_llm import LocalLLMClient

        readings = _store.get_recent_transformer_readings(transformer_id, n=48)
        score_data = _store.get_ml_score(transformer_id) or {}

        agent = DiagnosisAgent(kb=GridSenseKnowledgeBase(), llm=LocalLLMClient())
        wo_schema = agent.diagnose(
            transformer_id=transformer_id,
            anomaly_score=score_data.get("latest_score", 0.8),
            ewma_score=score_data.get("ewma_score", 0.8),
            hours_to_failure=score_data.get("hours_to_failure"),
            readings=readings,
            alert_id=work_order_id,
        )
        with _store._lock:
            for wo in _store.STORE["work_orders"]:
                if wo.get("id") == work_order_id:
                    wo.update(
                        {
                            "fault_type": wo_schema.fault_type,
                            "severity": wo_schema.severity,
                            "evidence": wo_schema.evidence,
                            "recommended_action": wo_schema.recommended_action,
                            "tools_needed": wo_schema.tools_needed,
                            "estimated_repair_hours": wo_schema.estimated_repair_hours,
                            "priority": wo_schema.priority.value,
                        }
                    )
                    break
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).error(
            "Background diagnosis failed for %s: %s", transformer_id, exc
        )


@router.put("/{work_order_id}/status")
async def update_work_order_status(work_order_id: str, body: dict) -> dict:
    """Update a work order status (PENDING → IN_PROGRESS → RESOLVED)."""
    status = body.get("status")
    valid = [s.value for s in WorkOrderStatus]
    if status not in valid:
        raise HTTPException(status_code=400, detail=f"status must be one of {valid}")
    updated = await repository.update_work_order_status(work_order_id, status)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Work order {work_order_id} not found")
    return {"id": work_order_id, "status": status, "updated": True}
