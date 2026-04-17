"""Object storage abstraction — local filesystem for dev, COS/OSS for prod.

All methods work with object keys like:
  papers/raw-pdf/Human_Object_Interaction/CVPR_2025/2025_InterMimic.pdf

The backend decides whether to store on local disk or cloud.
"""

import hashlib
import os
import shutil
from pathlib import Path

from backend.config import settings


class LocalStorage:
    """Local filesystem storage for development."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def put(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.write_bytes(data)
        return key

    async def put_file(self, key: str, local_path: str) -> str:
        path = self._resolve(key)
        shutil.copy2(local_path, path)
        return key

    async def get(self, key: str) -> bytes | None:
        path = self._resolve(key)
        if path.exists():
            return path.read_bytes()
        return None

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
        if path.exists():
            return path.stat().st_size
        return None

    def get_local_path(self, key: str) -> str | None:
        """For local storage, return the actual file path."""
        path = self._resolve(key)
        if path.exists():
            return str(path)
        return None


def compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum of data."""
    return hashlib.sha256(data).hexdigest()


def get_storage() -> LocalStorage:
    """Get the configured storage backend.

    For now, always returns LocalStorage.
    TODO: Add COS/OSS backends based on settings.object_storage_provider.
    """
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "storage"
    )
    return LocalStorage(base_dir)
