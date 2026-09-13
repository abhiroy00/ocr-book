"""
Upload validation & filesystem-safety helpers (spec section 5 & 32).

Every uploaded file goes through `validate_upload()` before anything is
written to disk or handed to PyMuPDF/Pillow. This never executes uploaded
content — it only inspects headers/metadata.
"""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

IMAGE_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class UploadValidationError(ValueError):
    """Raised for any rejected upload; message is safe to show to the user."""


@dataclass
class ValidatedUpload:
    extension: str
    mime_type: str
    size_bytes: int
    page_count: int
    is_pdf: bool


def sanitize_filename(filename: str) -> str:
    """Strip any path component and disallowed characters (path traversal guard).

    `os.path.basename()` only recognizes the *host OS's* separator -- on
    this Linux container that's `/` only, so a Windows-style upload name
    like `..\\..\\windows\\system32\\evil.pdf` (backslash is a perfectly
    legal filename character on POSIX) passed straight through unsplit,
    and each backslash-separated segment then got turned into its own
    `_`-joined chunk instead of being discarded as a directory component
    (confirmed by a real test failure: it produced
    `windows_system32_evil.pdf`, not `evil.pdf`). Normalizing `\\` to `/`
    first makes basename-extraction correct regardless of which
    separator style the original filename used or which OS this runs on.
    """
    normalized = (filename or "").replace("\\", "/")
    base = os.path.basename(normalized)
    safe = _SAFE_NAME_RE.sub("_", base).strip("._")
    return safe or f"upload_{uuid.uuid4().hex}"


def safe_join(root: str, *parts: str) -> str:
    """Join path segments under `root`, refusing to escape it."""
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, *parts))
    if not (target == root_abs or target.startswith(root_abs + os.sep)):
        raise UploadValidationError("Invalid path")
    return target


def validate_upload(filename: str, content: bytes) -> ValidatedUpload:
    settings = get_settings()

    ext = os.path.splitext(filename.lower())[1]
    if ext not in settings.allowed_extensions_list:
        raise UploadValidationError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(settings.allowed_extensions_list)}"
        )

    size_bytes = len(content)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size_bytes == 0:
        raise UploadValidationError("Uploaded file is empty")
    if size_bytes > max_bytes:
        raise UploadValidationError(f"File exceeds max upload size of {settings.max_upload_size_mb}MB")

    if ext == ".pdf":
        return _validate_pdf(content)
    return _validate_image(ext, content, size_bytes)


def _validate_pdf(content: bytes) -> ValidatedUpload:
    if not content.startswith(b"%PDF-"):
        raise UploadValidationError("File is not a valid PDF (bad header)")
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise UploadValidationError(f"Corrupted or unreadable PDF: {exc}") from exc

    try:
        if doc.is_encrypted:
            # Try an empty-password unlock (some scans are "encrypted" with no
            # real password, just permission flags); otherwise reject clearly.
            if not doc.authenticate(""):
                raise UploadValidationError(
                    "PDF is password protected. Please remove the password and re-upload."
                )
        page_count = doc.page_count
        if page_count == 0:
            raise UploadValidationError("PDF has no pages")
    finally:
        doc.close()

    return ValidatedUpload(extension=".pdf", mime_type="application/pdf", size_bytes=len(content), page_count=page_count, is_pdf=True)


def _validate_image(ext: str, content: bytes, size_bytes: int) -> ValidatedUpload:
    try:
        with Image.open(__import__("io").BytesIO(content)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise UploadValidationError(f"Corrupted or unreadable image: {exc}") from exc

    return ValidatedUpload(
        extension=ext,
        mime_type=IMAGE_MIME_BY_EXT.get(ext, "application/octet-stream"),
        size_bytes=size_bytes,
        page_count=1,
        is_pdf=False,
    )
