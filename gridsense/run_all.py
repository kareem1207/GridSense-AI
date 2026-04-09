"""GridSense AI — Orchestration entry point.

Starts all background services:
  - Checks llama-server availability
  - Checks MQTT broker availability
  - Starts MQTT ingestion consumer in a daemon thread
  - Starts the data simulator in a daemon thread
  - Starts the ML scoring loop in a daemon thread

Does NOT start FastAPI or Streamlit (user runs those manually).

Usage:
    uv run python gridsense/run_all.py
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

MQTT_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1")
ML_SCORE_INTERVAL = 30  # seconds between ML scoring passes

# Paths to saved ML model artifacts
_HERE = os.path.dirname(os.path.abspath(__file__))
IF_MODEL_PATH = os.path.join(_HERE, "ml", "models", "saved", "isolation_forest.joblib")
LSTM_MODEL_PATH = os.path.join(_HERE, "ml", "models", "saved", "lstm_autoencoder")


# ---------------------------------------------------------------------------
# Infrastructure checks
# ---------------------------------------------------------------------------

def check_llama_server() -> bool:
    """Return True if the local llama-server responds at LLM_URL."""
    try:
        import httpx

        base = LLM_URL.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        resp = httpx.get(f"{base}/health", timeout=2.0)
        return resp.status_code < 500
    except Exception:
        return False


def check_mqtt_broker() -> bool:
    """Return True if an MQTT broker is listening on MQTT_HOST:MQTT_PORT."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((MQTT_HOST, MQTT_PORT))
    sock.close()
    return result == 0


# ---------------------------------------------------------------------------
# Background service starters
# ---------------------------------------------------------------------------

def start_ingestion_consumer() -> threading.Thread:
    """Start the MQTT ingestion consumer in a daemon thread.

    Returns:
        The started Thread (already running)
    """
    from gridsense.ingestion.consumer import MQTTConsumer

    consumer = MQTTConsumer(broker_host=MQTT_HOST, broker_port=MQTT_PORT)

    def _run() -> None:
        consumer.start()
        logger.info("MQTT consumer started — subscribed to gridsense/#")
        # Keep thread alive
        while True:
            time.sleep(60)

    t = threading.Thread(target=_run, daemon=True, name="mqtt-consumer")
    t.start()
    return t


def start_simulator() -> threading.Thread:
    """Start the data simulator in a daemon thread.

    Returns:
        The started Thread (already running)
    """
    from gridsense.simulator.main import GridSimulator

    sim = GridSimulator(broker_host=MQTT_HOST, broker_port=MQTT_PORT)

    t = threading.Thread(target=sim.start, daemon=True, name="simulator")
    t.start()
    logger.info("Simulator started — 100 transformers / 5000 meters")
    return t


def start_ml_scoring_loop() -> threading.Thread:
    """Start the ML scoring loop in a daemon thread.

    Every ML_SCORE_INTERVAL seconds:
      1. Iterate all known transformer IDs.
      2. Run CombinedScorer.score() on recent readings.
      3. Write result to STORE["ml_scores"].
      4. If ewma_score > CRITICAL_THRESHOLD and no open work order exists,
         trigger DiagnosisAgent asynchronously.
      5. Create Alert record if score crosses WARNING or CRITICAL threshold.

    Returns:
        The started Thread (already running)
    """
    from gridsense.db import store
    from gridsense.ml.combined_scorer import (
        CRITICAL_THRESHOLD,
        WARNING_THRESHOLD,
        CombinedScorer,
    )

    # Load models
    scorer: Optional[CombinedScorer] = None
    if os.path.exists(IF_MODEL_PATH):
        scorer = CombinedScorer(
            if_model_path=IF_MODEL_PATH,
            lstm_model_path=LSTM_MODEL_PATH if os.path.exists(LSTM_MODEL_PATH + ".keras") else None,
        )
        logger.info("ML models loaded for scoring loop")
    else:
        logger.warning(
            "ML models not found at %s — run 'uv run python gridsense/ml/trainer.py' first. "
            "Scoring loop will start without ML until models appear.",
            IF_MODEL_PATH,
        )

    def _scoring_loop() -> None:
        nonlocal scorer

        while True:
            time.sleep(ML_SCORE_INTERVAL)

            # Lazy model load if trainer runs later
            if scorer is None and os.path.exists(IF_MODEL_PATH):
                try:
                    scorer = CombinedScorer(
                        if_model_path=IF_MODEL_PATH,
                        lstm_model_path=LSTM_MODEL_PATH
                        if os.path.exists(LSTM_MODEL_PATH + ".keras")
                        else None,
                    )
                    logger.info("ML models loaded lazily")
                except Exception as exc:
                    logger.warning("Lazy model load failed: %s", exc)
                    continue

            if scorer is None:
                continue

            transformer_ids = store.get_all_transformer_ids()
            if not transformer_ids:
                continue

            for tid in transformer_ids:
                try:
                    readings = store.get_recent_transformer_readings(tid, n=48)
                    if not readings:
                        continue

                    result = scorer.score(tid, readings)
                    score_dict = result.to_dict()
                    # Add score history to STORE
                    history = list(scorer._score_history.get(tid, []))
                    score_dict["score_history"] = history
                    store.update_ml_score(tid, score_dict)

                    # Create alert if crossing threshold for the first time
                    _maybe_create_alert(store, tid, result)

                    # Trigger AI diagnosis for critical transformers with no open WO
                    if result.ewma_score >= CRITICAL_THRESHOLD:
                        _maybe_trigger_diagnosis(store, tid, result, readings)

                except Exception as exc:
                    logger.error("Scoring failed for %s: %s", tid, exc)

    t = threading.Thread(target=_scoring_loop, daemon=True, name="ml-scorer")
    t.start()
    logger.info("ML scoring loop started (interval=%ds)", ML_SCORE_INTERVAL)
    return t


def _maybe_create_alert(store: object, transformer_id: str, result: object) -> None:
    """Create a WARNING or CRITICAL alert if the transformer just crossed a threshold."""
    import uuid
    from datetime import datetime, timezone

    from gridsense.ml.combined_scorer import CRITICAL_THRESHOLD, WARNING_THRESHOLD

    if result.alert_level == "NORMAL":
        return

    # Check if a recent active alert already exists for this transformer+level
    existing = [
        a for a in store.get_active_alerts()
        if a.get("transformer_id") == transformer_id
        and a.get("severity") == result.alert_level
    ]
    if existing:
        return  # Already alerted

    htf = result.hours_to_failure
    msg = (
        f"{result.alert_level} anomaly on {transformer_id}. "
        f"EWMA score: {result.ewma_score:.4f}. "
        f"{'Hours to failure: ' + str(htf) + 'h.' if htf is not None else 'Trend unknown.'}"
    )
    alert = {
        "id": str(uuid.uuid4()),
        "transformer_id": transformer_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": result.alert_level,
        "anomaly_score": result.raw_score,
        "ewma_score": result.ewma_score,
        "hours_to_failure": htf,
        "status": "ACTIVE",
        "message": msg,
    }
    store.add_alert(alert)
    logger.warning("ALERT created: %s on %s (score=%.4f)", result.alert_level, transformer_id, result.ewma_score)


def _maybe_trigger_diagnosis(
    store: object,
    transformer_id: str,
    result: object,
    readings: list,
) -> None:
    """Fire AI diagnosis if CRITICAL and no open work order exists."""
    import uuid
    from datetime import datetime, timezone

    existing_wos = store.get_work_orders_for_transformer(transformer_id)
    open_wos = [
        w for w in existing_wos
        if w.get("status") in ("PENDING", "IN_PROGRESS")
    ]
    if open_wos:
        return  # Already has an open work order

    logger.info("Triggering AI diagnosis for critical transformer %s", transformer_id)

    def _diagnose_async() -> None:
        try:
            from gridsense.genai.diagnosis_agent import DiagnosisAgent
            agent = DiagnosisAgent()
            wo_schema = agent.diagnose(
                transformer_id=transformer_id,
                anomaly_score=result.raw_score,
                ewma_score=result.ewma_score,
                hours_to_failure=result.hours_to_failure,
                readings=readings,
            )
            wo_dict = {
                "id": wo_schema.id,
                "transformer_id": wo_schema.transformer_id,
                "alert_id": wo_schema.alert_id,
                "created_at": wo_schema.created_at.isoformat(),
                "status": wo_schema.status.value,
                "fault_type": wo_schema.fault_type,
                "severity": wo_schema.severity,
                "evidence": wo_schema.evidence,
                "recommended_action": wo_schema.recommended_action,
                "tools_needed": wo_schema.tools_needed,
                "estimated_repair_hours": wo_schema.estimated_repair_hours,
                "priority": wo_schema.priority.value,
            }
            store.add_work_order(wo_dict)
            logger.info("Work order auto-created for %s: %s", transformer_id, wo_schema.id[:8])
        except Exception as exc:
            logger.error("Auto-diagnosis failed for %s: %s", transformer_id, exc)

    t = threading.Thread(target=_diagnose_async, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Print the GridSense AI startup banner."""
    banner = (
        "\n"
        "  +-------------------------------------------+\n"
        "  |         GridSense AI -- Starting...        |\n"
        "  |   Predictive Grid Fault Intelligence       |\n"
        "  +-------------------------------------------+\n"
        "\n"
        f"  Local LLM  : {LLM_URL}\n"
        "  API Server : http://127.0.0.1:8000\n"
        "  Dashboard  : http://localhost:8501\n"
        "  Simulator  : 100 transformers / 5000 meters\n"
    )
    print(banner)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Start all GridSense AI background services and block until Ctrl+C."""
    print_banner()

    # Checks
    llm_ok = check_llama_server()
    mqtt_ok = check_mqtt_broker()

    llm_status = "OK running" if llm_ok else "NOT running - GenAI will use templates"
    mqtt_status = "OK running" if mqtt_ok else "NOT running - start Mosquitto first"
    print(f"  LLM server  : {llm_status}")
    print(f"  MQTT broker : {mqtt_status}")
    print()

    if not mqtt_ok:
        print("  WARNING: MQTT broker not found on localhost:1883.")
        print("     Install Mosquitto: https://mosquitto.org/download/")
        print("     Then run: mosquitto -v")
        print()
        print("  Proceeding without MQTT - ML scoring loop will still run.")
        print()

    # Start background threads
    if mqtt_ok:
        start_ingestion_consumer()
        time.sleep(1)  # Let consumer connect before simulator publishes
        start_simulator()

    start_ml_scoring_loop()

    print("  Background services started.")
    print("  Start the API:       uvicorn gridsense.api.main:app --port 8000 --reload")
    print("  Start the dashboard: streamlit run gridsense/dashboard/app.py")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("GridSense AI shutting down...")
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
