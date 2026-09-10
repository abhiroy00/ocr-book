"""
Shared constants for the per-document Redis processing lock.

Split out from `app.workers.pipeline_tasks` (which owns acquire/renew/
release) so `app.services.document_service` can check whether a lock is
currently held without importing the worker module and creating a
circular import (pipeline_tasks already imports document_service).
"""
from __future__ import annotations

DOCUMENT_LOCK_PREFIX = "pipeline:lock:document:"
