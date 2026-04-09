"""Local LLM client — openai library pointed at llama-server (Gemma4 E4B)."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT: float = 30.0
LOCAL_LLM_URL: str = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1")
LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "gemma-4")


class LocalLLMClient:
    """OpenAI-compatible client for local llama-server running Gemma4 E4B.

    Falls back to deterministic template-based responses when the server is
    unavailable so that the rest of the system continues to function.

    Uses the openai Python library (v1.x) with a custom base_url.
    """

    def __init__(
        self,
        base_url: str = LOCAL_LLM_URL,
        model: str = LOCAL_LLM_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialise client configuration (no network IO here).

        Args:
            base_url: URL of the llama-server OpenAI-compatible endpoint
            model: model identifier passed to the server
            timeout: request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazily create the openai.OpenAI client."""
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key="not-needed",
                timeout=self.timeout,
            )
        return self._client

    def generate_diagnosis(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a diagnosis response from the local LLM.

        Args:
            system_prompt: role description and output format instructions
            user_prompt: query containing transformer sensor data and context
        Returns:
            Raw string response (ideally valid JSON), or template fallback.
        """
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(
                "LLM unavailable (%s: %s). Using fallback template.",
                type(exc).__name__,
                str(exc)[:120],
            )
            return self._fallback_response(user_prompt)

    def _fallback_response(self, user_prompt: str) -> str:
        """Return a deterministic JSON work-order template when the LLM is offline.

        Extracts ewma_score and transformer_id from the prompt text to tailor
        the response, so the system degrades gracefully without crashing.
        """
        score_match = re.search(
            r"ewma_smoothed_score[:\s]+([0-9.]+)", user_prompt, re.IGNORECASE
        )
        score = float(score_match.group(1)) if score_match else 0.85

        tid_match = re.search(
            r"transformer_id[:\s]+([A-Z0-9\-]+)", user_prompt, re.IGNORECASE
        )
        tid = tid_match.group(1) if tid_match else "UNKNOWN"

        if score >= 0.90:
            priority, severity = "HIGH", "CRITICAL"
            fault = "Transformer Overheating / Insulation Degradation"
            action = (
                "1. Dispatch field team immediately. "
                "2. Apply LOTO procedure. "
                "3. Emergency oil cooling or load shedding. "
                "4. Collect oil sample for DGA. "
                "5. Thermal imaging of all bushings."
            )
            hours = 4.0
        elif score >= 0.75:
            priority, severity = "MEDIUM", "WARNING"
            fault = "Elevated THD / Phase Imbalance"
            action = (
                "1. Schedule inspection within 48 hours. "
                "2. Perform load balancing assessment. "
                "3. Check capacitor bank and harmonic levels."
            )
            hours = 8.0
        else:
            priority, severity = "LOW", "NORMAL"
            fault = "Minor Parameter Anomaly"
            action = "Monitor remotely. Schedule routine maintenance during next patrol."
            hours = 2.0

        return json.dumps(
            {
                "fault_type": fault,
                "severity": severity,
                "evidence": (
                    f"Anomaly score {score:.3f} on transformer {tid}. "
                    "Automated template diagnosis (LLM offline)."
                ),
                "recommended_action": action,
                "tools_needed": "Thermal camera, oil test kit, multimeter, clamp meter",
                "estimated_repair_hours": hours,
                "priority": priority,
            }
        )

    def is_available(self) -> bool:
        """Return True if the local llama-server is reachable."""
        try:
            import httpx

            url = self.base_url.rstrip("/")
            if url.endswith("/v1"):
                url = url[:-3]
            resp = httpx.get(f"{url}/health", timeout=2.0)
            return resp.status_code < 500
        except Exception:
            return False
