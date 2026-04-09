"""RAG-augmented diagnosis agent for GridSense AI transformer fault analysis."""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from gridsense.genai.knowledge_base import GridSenseKnowledgeBase
from gridsense.genai.local_llm import LocalLLMClient
from gridsense.schemas.work_orders import WorkOrderPriority, WorkOrderSchema, WorkOrderStatus

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """<|think|>
You are GridSense AI, a senior power grid engineer and fault analyst at an Indian electricity distribution company (DISCOM). You have 20 years of experience diagnosing distribution transformer faults.

Your task: analyse the transformer sensor data provided and generate a structured work order for field engineers.

You MUST respond with ONLY a valid JSON object with EXACTLY these fields:
{
  "fault_type": "<specific fault type>",
  "severity": "<CRITICAL|WARNING|NORMAL>",
  "evidence": "<specific evidence citing actual sensor values>",
  "recommended_action": "<numbered actionable steps for the field engineer>",
  "tools_needed": "<comma-separated list of tools required>",
  "estimated_repair_hours": <float>,
  "priority": "<HIGH|MEDIUM|LOW>"
}

Rules:
- severity = CRITICAL when ewma_smoothed_score >= 0.90
- severity = WARNING when ewma_smoothed_score is 0.75 to 0.90
- priority HIGH for CRITICAL, MEDIUM for WARNING, LOW for NORMAL
- evidence MUST cite actual numbers from the sensor readings
- recommended_action MUST be specific numbered steps a field engineer can follow
- Do NOT output any text outside the JSON object
"""


class DiagnosisAgent:
    """RAG-augmented agent that diagnoses transformer faults and generates work orders.

    Retrieves relevant knowledge from ChromaDB, constructs a structured prompt,
    calls the local Gemma4 LLM, and parses the response into a WorkOrderSchema.
    Falls back to template-based diagnosis when the LLM is unavailable.
    """

    def __init__(
        self,
        kb: Optional[GridSenseKnowledgeBase] = None,
        llm: Optional[LocalLLMClient] = None,
    ) -> None:
        """Initialise the agent with a knowledge base and LLM client.

        Args:
            kb: GridSenseKnowledgeBase instance (created with defaults if None)
            llm: LocalLLMClient instance (created with defaults if None)
        """
        self._kb = kb or GridSenseKnowledgeBase()
        self._llm = llm or LocalLLMClient()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_query(self, ewma_score: float, readings: list[dict]) -> str:
        """Build a KB search query from the dominant anomaly characteristics."""
        if not readings:
            return "transformer anomaly high score fault detection"
        r = readings[-1]
        parts: list[str] = []
        if r.get("oil_temp", 60.0) > 80:
            parts.append("overheating oil temperature high")
        if r.get("thd_pct", 2.0) > 5:
            parts.append("high harmonic distortion THD")
        if r.get("power_factor", 0.92) < 0.80:
            parts.append("low power factor reactive power")
        ia = r.get("Ia", 50.0)
        ib = r.get("Ib", 50.0)
        ic = r.get("Ic", 50.0)
        avg_i = (ia + ib + ic) / 3 if (ia + ib + ic) > 0 else 1.0
        if (max(ia, ib, ic) - min(ia, ib, ic)) / avg_i > 0.3:
            parts.append("phase current imbalance fault")
        if ewma_score >= 0.90:
            parts.append("critical fault imminent failure insulation")
        elif ewma_score >= 0.75:
            parts.append("warning degradation overloading")
        return " ".join(parts) if parts else "transformer fault anomaly"

    def _format_readings_summary(self, readings: list[dict]) -> str:
        """Format the last 5 readings as a compact text table."""
        if not readings:
            return "No readings available."
        last5 = readings[-5:]
        lines = [
            "Recent DTM readings (last 5, oldest → newest):",
            f"{'Timestamp':<20} | {'Va':>6} | {'Ia':>6} | {'oil_°C':>6} | {'pf':>5} | {'thd%':>5} | {'kW':>6}",
            "-" * 70,
        ]
        for r in last5:
            ts = str(r.get("timestamp", ""))[:19]
            lines.append(
                f"{ts:<20} | {r.get('Va', 0):6.1f} | {r.get('Ia', 0):6.1f} | "
                f"{r.get('oil_temp', 0):6.1f} | {r.get('power_factor', 0):5.3f} | "
                f"{r.get('thd_pct', 0):5.2f} | {r.get('active_power_kw', 0):6.1f}"
            )
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        transformer_id: str,
        anomaly_score: float,
        ewma_score: float,
        hours_to_failure: Optional[float],
        readings: list[dict],
        rag_context: list[str],
    ) -> str:
        """Assemble the full user prompt with sensor data and KB context."""
        htf_str = f"{hours_to_failure:.1f} hours" if hours_to_failure is not None else "unknown"
        alert_level = (
            "CRITICAL" if ewma_score >= 0.90
            else "WARNING" if ewma_score >= 0.75
            else "NORMAL"
        )
        readings_table = self._format_readings_summary(readings)
        context_str = "\n\n---\n\n".join(rag_context) if rag_context else "No reference data."

        return (
            f"TRANSFORMER FAULT ANALYSIS REQUEST\n"
            f"transformer_id: {transformer_id}\n"
            f"anomaly_score: {anomaly_score:.4f}\n"
            f"ewma_smoothed_score: {ewma_score:.4f}\n"
            f"alert_level: {alert_level}\n"
            f"predicted_hours_to_failure: {htf_str}\n\n"
            f"{readings_table}\n\n"
            f"RELEVANT KNOWLEDGE BASE CONTEXT:\n{context_str}\n\n"
            f"Generate a work order JSON for this transformer fault."
        )

    def _parse_response(self, response: str) -> dict:
        """Extract and parse JSON from the LLM response with regex fallback.

        Handles markdown code blocks, leading/trailing text, and minor JSON errors.
        Returns a guaranteed-non-empty dict (uses hardcoded fallback on total failure).
        """
        stripped = response.strip()

        # Attempt 1: direct parse
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

        # Attempt 2: extract first {...} block
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Attempt 3: strip markdown code fence
        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM JSON response. Applying hardcoded fallback.")
        return {
            "fault_type": "Unknown Fault (Parse Error)",
            "severity": "WARNING",
            "evidence": "LLM response could not be parsed. Manual inspection required.",
            "recommended_action": "Send field engineer for physical inspection and manual assessment.",
            "tools_needed": "Thermal camera, multimeter, clamp meter",
            "estimated_repair_hours": 4.0,
            "priority": "MEDIUM",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        transformer_id: str,
        anomaly_score: float,
        ewma_score: float,
        hours_to_failure: Optional[float],
        readings: list[dict],
        alert_id: Optional[str] = None,
    ) -> WorkOrderSchema:
        """Diagnose a transformer fault and return a structured work order.

        Steps:
          1. Build KB search query from anomaly characteristics.
          2. Retrieve top-3 relevant knowledge chunks from ChromaDB.
          3. Construct structured user prompt.
          4. Call local LLM (Gemma4 via llama-server).
          5. Parse JSON response → WorkOrderSchema.

        Args:
            transformer_id: e.g. "T-047"
            anomaly_score: raw combined IF+LSTM score
            ewma_score: EWMA-smoothed score
            hours_to_failure: predicted hours to critical threshold (None if unknown)
            readings: list of recent DTM reading dicts (most recent last)
            alert_id: UUID of the triggering alert, for cross-linking
        Returns:
            WorkOrderSchema ready for storage and field dispatch
        """
        # Step 1-2: RAG retrieval
        query = self._build_query(ewma_score, readings)
        rag_context = self._kb.retrieve_similar_faults(query, top_k=3)

        # Step 3: Build prompt
        user_prompt = self._build_user_prompt(
            transformer_id, anomaly_score, ewma_score, hours_to_failure, readings, rag_context
        )

        # Step 4: LLM call (falls back to template if unavailable)
        raw_response = self._llm.generate_diagnosis(SYSTEM_PROMPT, user_prompt)

        # Step 5: Parse + map to schema
        data = self._parse_response(raw_response)
        priority_map: dict[str, WorkOrderPriority] = {
            "HIGH": WorkOrderPriority.HIGH,
            "MEDIUM": WorkOrderPriority.MEDIUM,
            "LOW": WorkOrderPriority.LOW,
        }
        priority = priority_map.get(
            str(data.get("priority", "MEDIUM")).upper(), WorkOrderPriority.MEDIUM
        )

        wo = WorkOrderSchema(
            id=str(uuid.uuid4()),
            transformer_id=transformer_id,
            alert_id=alert_id,
            created_at=datetime.now(timezone.utc),
            status=WorkOrderStatus.PENDING,
            fault_type=str(data.get("fault_type", "Unknown Fault")),
            severity=str(data.get("severity", "WARNING")),
            evidence=str(data.get("evidence", "")),
            recommended_action=str(data.get("recommended_action", "")),
            tools_needed=str(data.get("tools_needed", "")),
            estimated_repair_hours=float(data.get("estimated_repair_hours", 4.0)),
            priority=priority,
        )
        logger.info(
            "Work order %s created for %s (priority=%s, severity=%s)",
            wo.id[:8],
            transformer_id,
            priority.value,
            wo.severity,
        )
        return wo
