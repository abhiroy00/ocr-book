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
    page_parallelism: int = 4

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
