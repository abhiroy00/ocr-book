"""
Shared constants for the per-document Redis processing lock.

Split out from `app.workers.pipeline_tasks` (which owns acquire/renew/
release) so `app.services.document_service` can check whether a lock is
currently held without importing the worker module and creating a
circular import (pipeline_tasks already imports document_service).
"""
from __future__ import annotations

DOCUMENT_LOCK_PREFIX = "pipeline:lock:document:"

# Set by the `/cancel` API endpoint, checked cooperatively by the pipeline
# task between pages (see `pipeline_tasks._is_cancel_requested`). A plain
# Redis key rather than a DB column: cancellation is a transient signal
# the running task consumes and clears, not state anything else needs to
# query or migrate for.
DOCUMENT_CANCEL_PREFIX = "pipeline:cancel:document:"
