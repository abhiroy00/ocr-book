"""
Structured logging setup (structlog over stdlib logging).

Every pipeline stage should log through `get_stage_logger` so log lines are
uniformly greppable:

    [DOC-<id>] page=19 stage=OCR duration=2.31s status=ok confidence=0.94
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import structlog


def configure_logging(debug: bool = True) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
            if debug
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


@contextmanager
def stage_timer(
    logger: structlog.stdlib.BoundLogger,
    document_id: str,
    stage: str,
    page: int | None = None,
    **extra: Any,
) -> Iterator[dict]:
    """Times a pipeline stage and logs duration/status/error uniformly."""
    start = time.perf_counter()
    ctx: dict[str, Any] = {"status": "ok"}
    try:
        yield ctx
    except Exception as exc:  # noqa: BLE001 - re-raised after logging
        ctx["status"] = "error"
        ctx["error"] = str(exc)
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info(
            "pipeline_stage",
            document_id=document_id,
            page=page,
            stage=stage,
            duration=round(duration, 3),
            status=ctx.get("status"),
            error=ctx.get("error"),
            **extra,
        )
