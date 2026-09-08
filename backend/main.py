"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router, ws_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.app_debug)
logger = get_logger(__name__)

app = FastAPI(
    title="Document Clean & Reconstruct AI",
    description="Turns scanned PDFs/images into clean, searchable, editable documents while preserving original layout.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def on_startup() -> None:
    logger.info("app_startup", env=settings.app_env, ocr_provider=settings.ocr_provider.value, storage_backend=settings.storage_backend.value)
