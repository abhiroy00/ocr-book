"""
Centralized application configuration.

Every tunable in this file is read from environment variables (see
`.env.example`) via pydantic-settings. Nothing here is hardcoded to a
particular document, OCR provider, or deployment target.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRProviderName(str, Enum):
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"


class PreprocessProfile(str, Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    HIGH_QUALITY = "HIGH_QUALITY"


class StorageBackendName(str, Enum):
    LOCAL = "local"
    S3 = "s3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_env: str = "development"
    app_secret_key: str = "change-me"
    app_debug: bool = True
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- Database ---
    database_url: str = "postgresql+psycopg://docai:docai@localhost:5432/document_clean_reconstruct"

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # --- Storage ---
    storage_backend: StorageBackendName = StorageBackendName.LOCAL
    storage_local_root: str = "./storage"
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"

    # --- Upload ---
    max_upload_size_mb: int = 200
    allowed_upload_extensions: str = ".pdf,.jpg,.jpeg,.png,.webp"

    # --- Rendering / preprocessing ---
    default_dpi: int = 300
    allowed_dpis: str = "150,200,300,400,600"
    preprocess_profile: PreprocessProfile = PreprocessProfile.BALANCED

    # --- OCR ---
    ocr_provider: OCRProviderName = OCRProviderName.PADDLEOCR
    paddle_ocr_langs: str = "en,hi"
    tesseract_path: str | None = None
    tesseract_langs: str = "eng+hin"

    nvidia_api_key: str | None = None
    nvidia_model: str = "nvidia/neva-22b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # PP-Structure (PaddleOCR's table-structure model) is an optional,
    # best-effort enhancement over the OpenCV-line + OCR-alignment table
    # engines (which are the fully-supported default path). Its native
    # inference has been observed to hard-crash the process (SIGILL) on
    # some CPU/virtualization combinations where the prebuilt paddlepaddle
    # wheel doesn't actually match the host's available instruction set --
    # a crash `try/except` cannot catch and that takes the whole Celery
    # worker process down. Off by default; enable only after confirming
    # PaddleOCR itself runs cleanly in your environment.
    enable_pp_structure: bool = False

    # --- Workers ---
    celery_worker_concurrency: int = 4

    # --- Page-level OCR parallelism (controlled worker pool, not one
    # process per page -- see app.workers.page_worker_pool) ---
    # 0 means "pick a sensible default from CPU count at pipeline start"
    # (see Settings.resolved_ocr_workers) rather than hardcoding a number
    # here that would be wrong on both a 2-core laptop and a 32-core
    # server. PaddleOCR/OpenCV work is CPU-bound native code with no GIL
    # release, so this pool is process-based, not thread-based -- each
    # worker's memory cost (a loaded OCR model) is real, so this must stay
    # a deliberate, bounded number, never one-per-page.
    ocr_workers: int = 0
    # Upper ceiling used when OCR_WORKERS is left at its auto default (0)
    # -- the literal throughput target, e.g. 10 on a 14-core host. See
    # `resolved_ocr_workers`: this is a REQUEST, not a guarantee, since
    # actual available RAM at pipeline start might not support it.
    ocr_max_workers: int = 10
    ocr_min_workers: int = 1
    # CPU cores held back for Postgres/Redis/the web server/other Celery
    # concurrency, never handed to the OCR pool.
    ocr_reserved_cpu: int = 3
    # RAM held back (MB) for everything else sharing this host (Postgres,
    # Redis, FastAPI, the Celery worker's own base footprint, OS/Docker
    # overhead) before computing how many OCR workers fit. 1536MB matches
    # the measured ~424MB idle celery-worker baseline plus real margin for
    # the other services.
    ocr_reserved_memory_mb: int = 1536
    # Measured, not guessed -- and revised upward after a first, INCOMPLETE
    # measurement undercounted this. On 2026-09-13: a 4-worker run crashed
    # a 7.5GB VM at 6.4GB after only ~2 pages/worker (interrupted before
    # memory finished climbing, implying ~1.5GB/worker -- too low). A
    # longer, uninterrupted follow-up run (2 workers, 10 full pages, 12GB
    # VM) reached 9.57GB before plateauing -- (9.57GB - 0.48GB idle
    # baseline) / 2 workers = ~4.5GB/worker at sustained, warmed-up
    # steady state (PaddleOCR with en+hi both loaded per worker -- see
    # PADDLE_OCR_LANGS). Rounded up to 4800MB for margin. Override this if
    # a different PADDLE_OCR_LANGS configuration changes the real
    # per-worker footprint on your host -- and prefer a longer measurement
    # (10+ pages) over a short one: early pages undercount the true cost.
    ocr_worker_est_memory_mb: int = 4800
    # False bypasses the RAM-aware calculation below entirely and honors
    # OCR_WORKERS literally -- an explicit escape hatch for an operator
    # who has already sized their own host. The default (True) is what
    # actually prevents "blindly start N workers and OOM" -- the exact
    # failure mode that crashed this host during testing.
    ocr_auto_fallback: bool = True
    # Recycle (restart) each page-worker process after this many pages,
    # even mid-document -- confirmed necessary by direct measurement
    # (2026-09-14): a single worker's memory grew unbounded over a long
    # run (8GB after 68 pages on a real 446-page document, vs. ~4.5GB
    # after 10 in a short benchmark), degrading per-page OCR time roughly
    # 10x alongside it (171s/page vs. a clean ~15-17s/page baseline) --
    # consistent with memory-pressure thrashing, not a one-time model-load
    # cost. See `app.workers.page_worker_pool.PageWorkerPool`. 25 is a
    # practical balance: frequent enough to keep memory from compounding
    # for hours, infrequent enough that the model-reload cost on respawn
    # (a few seconds) stays a small fraction of total time.
    ocr_worker_max_pages: int = 25
    # Independent, much smaller concurrency cap for NVIDIA AI-fallback
    # calls specifically -- normal OCR parallelism (OCR_WORKERS) must not
    # translate into that many simultaneous external API requests.
    max_ai_concurrency: int = 2
    ocr_queue_size: int = 0  # 0 = unbounded (page numbers are cheap; no reason to block submission)
    ocr_task_timeout_seconds: int = 180
    ocr_retry_count: int = 2

    # --- Library accession register (PDF -> DB -> cumulative Master Excel) ---
    # Auto-generated accession numbers look like "{prefix}-{n}" (e.g.
    # "D-305"). Configurable because this system may be continuing an
    # existing physical/manual register that was already at some number
    # when automation started -- there is no way to derive that starting
    # point from the documents themselves, so it must be set explicitly
    # rather than always starting at 1.
    accession_number_prefix: str = "D"
    accession_number_start: int = 1

    # --- Master Accession Register (persisted, incrementally-appended
    # workbook -- app.services.accession_register_service) ---
    # How long a worker waits for another worker's write (load -> append/
    # update -> sort -> atomic save) to finish before giving up rather
    # than blocking forever on a peer that may itself be stuck. On
    # timeout the append is logged and skipped for that document, not
    # retried automatically -- the DB-backed AccessionRecord it reads from
    # remains the source of truth, so nothing is lost, just not yet
    # reflected in the on-disk workbook.
    master_register_lock_timeout_seconds: float = 30.0

    @field_validator("allowed_upload_extensions")
    @classmethod
    def _normalize_ext(cls, v: str) -> str:
        return v.lower()

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [e.strip() for e in self.allowed_upload_extensions.split(",") if e.strip()]

    @property
    def allowed_dpi_list(self) -> List[int]:
        return [int(d) for d in self.allowed_dpis.split(",") if d.strip()]

    @property
    def paddle_ocr_lang_list(self) -> List[str]:
        return [l.strip() for l in self.paddle_ocr_langs.split(",") if l.strip()]

    @property
    def resolved_ocr_workers(self) -> int:
        """Safe OCR worker-pool size, computed fresh at pipeline start --
        bounded by CPU headroom, by *actually measured available RAM*
        (via psutil), and by a configurable ceiling (OCR_MAX_WORKERS),
        never by a single trusted number alone.

        Why RAM matters as much as CPU: confirmed by direct measurement
        (2026-09-13) that CPU was never this host's real constraint (14
        cores, mostly idle at OCR_WORKERS=1) -- RAM was. 4 real OCR
        workers pushed a 7.5GB Docker Desktop VM to 6.4GB and crashed the
        Docker engine itself. Each worker process loads its own PaddleOCR
        model(s) (real, non-shareable memory, not just CPU time), so
        "request 10 workers" must be clamped by how many actually fit in
        whatever RAM is free right now -- which can shift between runs
        (another job, another service) -- not assumed constant.

        `OCR_WORKERS` (legacy) still works as the requested ceiling when
        set > 0; `OCR_MAX_WORKERS` is the clearer newer name for the same
        thing. Sizing is min(requested, cpu_budget, ram_budget), floored
        at OCR_MIN_WORKERS so a job can always make forward progress.
        Set OCR_AUTO_FALLBACK=false to skip the RAM check and honor
        OCR_WORKERS literally -- an explicit escape hatch, not the
        default, since bypassing this is exactly the "blindly start N
        workers" failure mode that crashed this host during testing.
        """
        import os

        requested = self.ocr_workers if self.ocr_workers > 0 else self.ocr_max_workers

        if not self.ocr_auto_fallback:
            return max(1, requested)

        cpu_count = os.cpu_count() or 2
        cpu_budget = max(1, cpu_count - self.ocr_reserved_cpu)

        try:
            import psutil

            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:  # noqa: BLE001 - psutil missing/failed: degrade to the old CPU-only heuristic, never crash config loading over this
            available_mb = None

        if available_mb is None:
            ram_budget = max(1, min(6, cpu_count // 2))
        else:
            usable_mb = max(0.0, available_mb - self.ocr_reserved_memory_mb)
            ram_budget = max(1, int(usable_mb // self.ocr_worker_est_memory_mb))

        safe = min(requested, cpu_budget, ram_budget)
        return max(self.ocr_min_workers, safe)

    @property
    def ocr_worker_sizing_debug(self) -> dict:
        """The full breakdown behind `resolved_ocr_workers`, for logging
        at pipeline start (section 15: observability) -- so "why did we
        get N workers, not the requested ceiling" is answerable from logs
        alone rather than requiring a live investigation each time."""
        import os

        requested = self.ocr_workers if self.ocr_workers > 0 else self.ocr_max_workers
        cpu_count = os.cpu_count() or 2
        cpu_budget = max(1, cpu_count - self.ocr_reserved_cpu)

        try:
            import psutil

            available_mb = round(psutil.virtual_memory().available / (1024 * 1024))
        except Exception:  # noqa: BLE001
            available_mb = None

        ram_budget = (
            max(1, min(6, cpu_count // 2))
            if available_mb is None
            else max(1, int(max(0, available_mb - self.ocr_reserved_memory_mb) // self.ocr_worker_est_memory_mb))
        )
        chosen = self.resolved_ocr_workers
        limiting_factor = "auto_fallback_disabled" if not self.ocr_auto_fallback else (
            "requested" if chosen == requested else ("cpu" if chosen == cpu_budget else "ram")
        )
        return {
            "requested": requested,
            "cpu_count": cpu_count,
            "cpu_budget": cpu_budget,
            "available_ram_mb": available_mb,
            "ram_budget": ram_budget,
            "chosen": chosen,
            "limiting_factor": limiting_factor,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
