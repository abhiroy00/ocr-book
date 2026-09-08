"""
Pluggable storage backend: local filesystem (development) or S3-compatible
object storage (production), selected by `STORAGE_BACKEND`.

Nothing else in the codebase should call `open()`/`os.path` directly against
document assets — always go through `get_storage()`.
"""
from __future__ import annotations

import abc
import os
import shutil
from functools import lru_cache

from app.core.config import StorageBackendName, get_settings
from app.utils.file_safety import safe_join


class StorageBackend(abc.ABC):
    @abc.abstractmethod
    def write(self, relative_path: str, data: bytes) -> str: ...

    @abc.abstractmethod
    def read(self, relative_path: str) -> bytes: ...

    @abc.abstractmethod
    def exists(self, relative_path: str) -> bool: ...

    @abc.abstractmethod
    def delete(self, relative_path: str) -> None: ...

    @abc.abstractmethod
    def delete_prefix(self, relative_prefix: str) -> None: ...

    @abc.abstractmethod
    def local_path(self, relative_path: str) -> str:
        """Returns (creating parent dirs if needed for writes) a local filesystem
        path a native library (PyMuPDF, OpenCV, python-docx) can read/write
        directly. For S3, this materializes to/from a local temp cache."""

    @abc.abstractmethod
    def url_for(self, relative_path: str) -> str:
        """Public/servable URL (or API-relative path) for the frontend."""

    def sync_from_local(self, relative_path: str) -> None:
        """Call after writing directly to `local_path(relative_path)` (e.g. via
        cv2.imwrite / PyMuPDF / python-docx .save()) so backends that need an
        explicit upload (S3) persist the result. No-op for local storage."""
        return None


class LocalStorageBackend(StorageBackend):
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _abs(self, relative_path: str) -> str:
        return safe_join(self.root, relative_path)

    def write(self, relative_path: str, data: bytes) -> str:
        path = self._abs(relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return relative_path

    def read(self, relative_path: str) -> bytes:
        with open(self._abs(relative_path), "rb") as f:
            return f.read()

    def exists(self, relative_path: str) -> bool:
        return os.path.exists(self._abs(relative_path))

    def delete(self, relative_path: str) -> None:
        path = self._abs(relative_path)
        if os.path.exists(path):
            os.remove(path)

    def delete_prefix(self, relative_prefix: str) -> None:
        path = self._abs(relative_prefix)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

    def local_path(self, relative_path: str) -> str:
        path = self._abs(relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def url_for(self, relative_path: str) -> str:
        return f"/api/files/{relative_path.replace(os.sep, '/')}"


class S3StorageBackend(StorageBackend):
    def __init__(self, bucket: str, endpoint_url: str | None, access_key: str | None, secret_key: str | None, region: str) -> None:
        import boto3  # local import: keep boto3 optional for local-only dev

        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            region_name=region,
        )
        self._tmp_root = "/tmp/docai_s3_cache"
        os.makedirs(self._tmp_root, exist_ok=True)

    def write(self, relative_path: str, data: bytes) -> str:
        self._client.put_object(Bucket=self.bucket, Key=relative_path, Body=data)
        return relative_path

    def read(self, relative_path: str) -> bytes:
        obj = self._client.get_object(Bucket=self.bucket, Key=relative_path)
        return obj["Body"].read()

    def exists(self, relative_path: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=relative_path)
            return True
        except ClientError:
            return False

    def delete(self, relative_path: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=relative_path)

    def delete_prefix(self, relative_prefix: str) -> None:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=relative_prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:
                self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})

    def local_path(self, relative_path: str) -> str:
        local = safe_join(self._tmp_root, relative_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if self.exists(relative_path) and not os.path.exists(local):
            with open(local, "wb") as f:
                f.write(self.read(relative_path))
        return local

    def url_for(self, relative_path: str) -> str:
        return self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": relative_path}, ExpiresIn=3600
        )

    def sync_from_local(self, relative_path: str) -> None:
        local = safe_join(self._tmp_root, relative_path)
        if os.path.exists(local):
            with open(local, "rb") as f:
                self.write(relative_path, f.read())


@lru_cache
def get_storage() -> StorageBackend:
    settings = get_settings()
    if settings.storage_backend == StorageBackendName.S3:
        if not settings.s3_bucket:
            raise RuntimeError("STORAGE_BACKEND=s3 requires S3_BUCKET to be set")
        return S3StorageBackend(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region=settings.s3_region,
        )
    return LocalStorageBackend(settings.storage_local_root)
