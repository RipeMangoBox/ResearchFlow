"""Object storage abstraction — local filesystem for dev, COS for prod.

All methods work with object keys like:
  papers/raw-pdf/Human_Object_Interaction/CVPR_2025/2025_InterMimic.pdf

Auto-selects backend based on settings.object_storage_provider:
  "local" → LocalStorage (filesystem, for dev)
  "cos"   → COSStorage (Tencent Cloud COS, for prod)
"""

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)


# ── Interface / Base ────────────────────────────────────────────

class StorageBackend:
    """Common interface for all storage backends."""

    async def put(self, key: str, data: bytes) -> str:
        raise NotImplementedError

    async def put_file(self, key: str, local_path: str) -> str:
        raise NotImplementedError

    async def get(self, key: str) -> bytes | None:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        raise NotImplementedError

    async def get_size(self, key: str) -> int | None:
        raise NotImplementedError

    def get_local_path(self, key: str) -> str | None:
        """Return local file path if available (for PDF parsing)."""
        return None


# ── Local Storage ───────────────────────────────────────────────

class LocalStorage(StorageBackend):
    """Local filesystem storage for development."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def put(self, key: str, data: bytes) -> str:
        self._resolve(key).write_bytes(data)
        return key

    async def put_file(self, key: str, local_path: str) -> str:
        shutil.copy2(local_path, self._resolve(key))
        return key

    async def get(self, key: str) -> bytes | None:
        path = self._resolve(key)
        return path.read_bytes() if path.exists() else None

    async def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    async def delete(self, key: str) -> bool:
        path = self._resolve(key)
        if path.exists():
            path.unlink()
            return True
        return False

    async def get_size(self, key: str) -> int | None:
        path = self._resolve(key)
        return path.stat().st_size if path.exists() else None

    def get_local_path(self, key: str) -> str | None:
        path = self._resolve(key)
        return str(path) if path.exists() else None


# ── Tencent Cloud COS Storage ──────────────────────────────────

class COSStorage(StorageBackend):
    """Tencent Cloud COS storage for production.

    Requires: pip install cos-python-sdk-v5
    Config: OBJECT_STORAGE_SECRET_ID, SECRET_KEY, REGION, BUCKET
    """

    def __init__(self):
        from qcloud_cos import CosConfig, CosS3Client

        config = CosConfig(
            Region=settings.object_storage_region,
            SecretId=settings.object_storage_secret_id,
            SecretKey=settings.object_storage_secret_key,
        )
        self.client = CosS3Client(config)
        self.bucket = settings.object_storage_bucket
        # Local cache for downloaded files (PDF parsing needs local path)
        self._cache_dir = Path(tempfile.gettempdir()) / "rf_cos_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"COS storage initialized: bucket={self.bucket}, region={settings.object_storage_region}")

    async def put(self, key: str, data: bytes) -> str:
        self.client.put_object(
            Bucket=self.bucket,
            Body=data,
            Key=key,
        )
        return key

    async def put_file(self, key: str, local_path: str) -> str:
        self.client.upload_file(
            Bucket=self.bucket,
            Key=key,
            LocalFilePath=local_path,
        )
        return key

    async def get(self, key: str) -> bytes | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].get_raw_stream().read()
        except Exception as e:
            if "NoSuchKey" in str(e):
                return None
            raise

    async def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    async def get_size(self, key: str) -> int | None:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            return int(response.get("Content-Length", 0))
        except Exception:
            return None

    def get_local_path(self, key: str) -> str | None:
        """Download to local cache and return path (for PDF parsing)."""
        cache_path = self._cache_dir / key.replace("/", "_")
        if cache_path.exists():
            return str(cache_path)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            data = response["Body"].get_raw_stream().read()
            cache_path.write_bytes(data)
            return str(cache_path)
        except Exception as e:
            logger.warning(f"COS download failed for {key}: {e}")
            return None


# ── Factory ─────────────────────────────────────────────────────

_storage_instance: StorageBackend | None = None


def compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum of data."""
    return hashlib.sha256(data).hexdigest()


def get_storage() -> StorageBackend:
    """Get the configured storage backend (singleton).

    Selection:
      - "cos" + valid secret_id → COSStorage
      - otherwise → LocalStorage (development fallback)
    """
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    provider = settings.object_storage_provider.lower()

    if provider == "cos" and settings.object_storage_secret_id:
        try:
            _storage_instance = COSStorage()
            return _storage_instance
        except Exception as e:
            logger.warning(f"COS init failed, falling back to local: {e}")

    # Fallback: local storage
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "storage",
    )
    _storage_instance = LocalStorage(base_dir)
    logger.info(f"Using local storage: {base_dir}")
    return _storage_instance
