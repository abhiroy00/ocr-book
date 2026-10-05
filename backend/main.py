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
def health():
    """Load-balancer health check (AWS ALB target group / nginx). Reports
    unhealthy (503) when this replica cannot reach Postgres or Redis --
    the two things every real request needs -- so the ALB stops routing to
    it, instead of only proving the process is up. Kept cheap (SELECT 1 +
    PING, short timeouts): it runs every few seconds per target."""
    import redis as redis_sync
    from fastapi.responses import JSONResponse
    from sqlalchemy import text

    from app.db.session import engine

    checks: dict[str, str] = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 - any failure means unhealthy, never a 500
        logger.warning("health_database_failed", error=str(exc))
        checks["database"] = "unavailable"
    try:
        client = redis_sync.Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
        try:
            client.ping()
        finally:
            client.close()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("health_redis_failed", error=str(exc))
        checks["redis"] = "unavailable"

    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(status_code=200 if healthy else 503, content={"status": "ok" if healthy else "unhealthy", **checks})


@app.on_event("startup")
def on_startup() -> None:
    logger.info("app_startup", env=settings.app_env, ocr_provider=settings.ocr_provider.value, storage_backend=settings.storage_backend.value)
