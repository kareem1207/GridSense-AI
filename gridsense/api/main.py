"""GridSense AI — FastAPI application entry point."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gridsense.api.routes import alerts, meters, system, transformers, work_orders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

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
    """Initialise the GenAI knowledge base on API startup."""
    import asyncio

    logger.info("GridSense AI API starting up…")
    try:
        from gridsense.genai.knowledge_base import GridSenseKnowledgeBase

        kb = GridSenseKnowledgeBase()
        if not kb.is_built():
            await asyncio.to_thread(kb.build)
            logger.info("Knowledge base initialised")
        else:
            logger.info("Knowledge base already built — skipping")
    except Exception as exc:
        logger.warning("Knowledge base init failed (non-fatal): %s", exc)
