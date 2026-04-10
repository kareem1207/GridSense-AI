"""GridSense AI FastAPI application entry point."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gridsense.api.routes import alerts, meters, system, transformers, work_orders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "true") -> bool:
    """Parse a boolean-like environment variable."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


app = FastAPI(
    title="GridSense AI",
    description="Predictive Grid Fault Intelligence API for Indian DISCOMs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route modules
app.include_router(system.router)
app.include_router(transformers.router)
app.include_router(alerts.router)
app.include_router(work_orders.router)
app.include_router(meters.router)


@app.on_event("startup")
async def on_startup() -> None:
    """Initialise optional background services and the GenAI knowledge base."""
    import asyncio

    logger.info("GridSense AI API starting up...")
    app.state.mqtt_consumer = None
    app.state.ml_thread = None

    if _env_flag("GRIDSENSE_START_INGESTION_ON_API_STARTUP", "true"):
        try:
            from gridsense.ingestion.consumer import start_consumer

            app.state.mqtt_consumer = start_consumer()
            logger.info(
                "Embedded MQTT ingestion consumer started inside API process. "
                "Avoid running a separate consumer at the same time, or readings will be duplicated."
            )
        except Exception as exc:
            logger.warning("Embedded MQTT consumer failed to start (non-fatal): %s", exc)
    else:
        logger.info("Embedded MQTT consumer disabled by environment")

    if _env_flag("GRIDSENSE_START_ML_ON_API_STARTUP", "true"):
        try:
            from gridsense.run_all import start_ml_scoring_loop

            app.state.ml_thread = start_ml_scoring_loop()
            logger.info("Embedded ML scoring loop started inside API process")
        except Exception as exc:
            logger.warning("Embedded ML scoring loop failed to start (non-fatal): %s", exc)
    else:
        logger.info("Embedded ML scoring loop disabled by environment")

    try:
        from gridsense.genai.knowledge_base import GridSenseKnowledgeBase

        kb = GridSenseKnowledgeBase()
        if not kb.is_built():
            await asyncio.to_thread(kb.build)
            logger.info("Knowledge base initialised")
        else:
            logger.info("Knowledge base already built - skipping")
    except Exception as exc:
        logger.warning("Knowledge base init failed (non-fatal): %s", exc)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Stop embedded background services cleanly."""
    consumer = getattr(app.state, "mqtt_consumer", None)
    if consumer is not None:
        try:
            consumer.stop()
        except Exception as exc:
            logger.warning("Embedded MQTT consumer shutdown failed: %s", exc)
