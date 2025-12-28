"""Storage service implementation."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from grpc import aio as grpc_aio

from aiglasses.common import BaseService, get_logger
from aiglasses.common.events import Event, get_event_bus
from aiglasses.config import Config


@dataclass
class StorageItem:
    """Stored item metadata."""

    key: str
    data: bytes
    content_type: str
    category: str
    app_id: str | None
    created_at: float
    expires_at: float | None
    metadata: dict = field(default_factory=dict)


@dataclass
class EphemeralHandle:
    """Ephemeral storage handle."""

    handle_id: str
    path: str
    expires_at: float


@dataclass
class RetentionPolicy:
    """Data retention policy."""

    default_retention_seconds: int = 60
    audio_retention_seconds: int = 30
    video_retention_seconds: int = 30
    frame_retention_seconds: int = 60
    text_retention_seconds: int = 300
    max_storage_bytes: int = 500 * 1024 * 1024  # 500MB
    encrypt_at_rest: bool = False
    secure_delete: bool = True


class StorageService(BaseService):
    """Storage service.

    Responsibilities:
    - Privacy-aware data storage
    - Retention policy enforcement
    - Ephemeral storage
    """

    def __init__(self, config: Config | None = None, mock_mode: bool = False) -> None:
        super().__init__("storage", config, mock_mode)
        self._items: dict[str, StorageItem] = {}
        self._ephemeral_handles: dict[str, EphemeralHandle] = {}
        self._policy = RetentionPolicy()
        self._event_bus = get_event_bus()
        self._cleanup_task: asyncio.Task | None = None
        self._storage_dir = Path(tempfile.gettempdir()) / "aiglasses_storage"

    @property
    def port(self) -> int:
        return self.config.ports.storage

    async def setup(self) -> None:
        """Setup Storage service."""
        self.logger.info("storage_service_setup", mock_mode=self.mock_mode)

        # Apply config to policy
        self._policy.default_retention_seconds = self.config.privacy.retention_seconds
        self._policy.audio_retention_seconds = self.config.privacy.audio_retention_seconds
        self._policy.video_retention_seconds = self.config.privacy.video_retention_seconds
        self._policy.frame_retention_seconds = self.config.privacy.frame_retention_seconds
        self._policy.encrypt_at_rest = self.config.privacy.encrypt_storage
        self._policy.secure_delete = self.config.privacy.secure_delete

        # Create storage directory
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def teardown(self) -> None:
        """Teardown Storage service."""
        self.logger.info("storage_service_teardown")

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Purge all data
        await self.purge()

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services."""
        from aiglasses.foundation.storage.grpc_servicer import StorageServicer, add_servicer

        servicer = StorageServicer(self)
        add_servicer(server, servicer)

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired items."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                now = time.time()
                expired_keys = [
                    key
                    for key, item in self._items.items()
                    if item.expires_at and item.expires_at < now
                ]

                for key in expired_keys:
                    await self.delete(key)
                    self.logger.debug("item_expired", key=key)

                # Check ephemeral handles
                expired_handles = [
                    h_id
                    for h_id, handle in self._ephemeral_handles.items()
                    if handle.expires_at < now
                ]

                for h_id in expired_handles:
                    await self.release_ephemeral(h_id)
                    self.logger.debug("ephemeral_expired", handle_id=h_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("cleanup_error", error=str(e))

    def _get_retention(self, category: str) -> int:
        """Get retention seconds for a category."""
        return {
            "audio": self._policy.audio_retention_seconds,
            "video": self._policy.video_retention_seconds,
            "frame": self._policy.frame_retention_seconds,
            "text": self._policy.text_retention_seconds,
        }.get(category, self._policy.default_retention_seconds)

    async def store(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        category: str = "other",
        app_id: str | None = None,
        retention_seconds: int | None = None,
        metadata: dict | None = None,
    ) -> StorageItem:
        """Store data.

        Args:
            key: Storage key.
            data: Data bytes.
            content_type: Content type.
            category: Data category.
            app_id: Owning app.
            retention_seconds: Retention override.
            metadata: Additional metadata.

        Returns:
            Storage item.
        """
        now = time.time()
        retention = retention_seconds or self._get_retention(category)
        expires_at = now + retention if retention > 0 else None

        item = StorageItem(
            key=key,
            data=data,
            content_type=content_type,
            category=category,
            app_id=app_id,
            created_at=now,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        self._items[key] = item

        await self._event_bus.publish(
            Event(
                topic="storage.stored",
                data={"key": key, "category": category, "size": len(data)},
                source="storage",
            )
        )

        return item

    async def retrieve(self, key: str) -> StorageItem | None:
        """Retrieve data by key.

        Args:
            key: Storage key.

        Returns:
            Storage item or None.
        """
        item = self._items.get(key)
        if item and item.expires_at and item.expires_at < time.time():
            await self.delete(key)
            return None
        return item

    async def delete(self, key: str, secure: bool = False) -> bool:
        """Delete data by key.

        Args:
            key: Storage key.
            secure: Overwrite with zeros first.

        Returns:
            True if deleted.
        """
        if key not in self._items:
            return False

        item = self._items[key]
        if secure or self._policy.secure_delete:
            # Overwrite data with zeros (in-memory secure delete)
            item.data = bytes(len(item.data))

        del self._items[key]
        return True

    async def list_items(
        self,
        prefix: str | None = None,
        category: str | None = None,
        app_id: str | None = None,
        limit: int = 100,
    ) -> list[StorageItem]:
        """List stored items.

        Args:
            prefix: Key prefix filter.
            category: Category filter.
            app_id: App filter.
            limit: Maximum items.

        Returns:
            List of items (without data).
        """
        items = []
        for key, item in self._items.items():
            if prefix and not key.startswith(prefix):
                continue
            if category and item.category != category:
                continue
            if app_id and item.app_id != app_id:
                continue
            if item.expires_at and item.expires_at < time.time():
                continue
            items.append(item)
            if len(items) >= limit:
                break
        return items

    async def create_ephemeral(
        self,
        ttl_seconds: int = 60,
        max_size_bytes: int = 10 * 1024 * 1024,
    ) -> EphemeralHandle:
        """Create ephemeral storage handle.

        Args:
            ttl_seconds: Time to live.
            max_size_bytes: Maximum size.

        Returns:
            Ephemeral handle.
        """
        handle_id = str(uuid.uuid4())
        path = str(self._storage_dir / handle_id)

        # Create temp file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).touch()

        handle = EphemeralHandle(
            handle_id=handle_id,
            path=path,
            expires_at=time.time() + ttl_seconds,
        )
        self._ephemeral_handles[handle_id] = handle

        return handle

    async def release_ephemeral(self, handle_id: str) -> None:
        """Release ephemeral storage handle.

        Args:
            handle_id: Handle ID.
        """
        if handle_id not in self._ephemeral_handles:
            return

        handle = self._ephemeral_handles[handle_id]

        # Securely delete file
        try:
            if os.path.exists(handle.path):
                if self._policy.secure_delete:
                    with open(handle.path, "r+b") as f:
                        size = f.seek(0, 2)
                        f.seek(0)
                        f.write(bytes(size))
                os.remove(handle.path)
        except Exception as e:
            self.logger.warning("ephemeral_delete_failed", handle_id=handle_id, error=str(e))

        del self._ephemeral_handles[handle_id]

    async def purge(
        self,
        category: str | None = None,
        app_id: str | None = None,
    ) -> tuple[int, int]:
        """Purge all data.

        Args:
            category: Filter by category.
            app_id: Filter by app.

        Returns:
            Tuple of (items_deleted, bytes_freed).
        """
        items_deleted = 0
        bytes_freed = 0

        keys_to_delete = []
        for key, item in self._items.items():
            if category and item.category != category:
                continue
            if app_id and item.app_id != app_id:
                continue
            keys_to_delete.append(key)
            bytes_freed += len(item.data)

        for key in keys_to_delete:
            await self.delete(key, secure=True)
            items_deleted += 1

        await self._event_bus.publish(
            Event(
                topic="storage.purged",
                data={"items_deleted": items_deleted, "bytes_freed": bytes_freed},
                source="storage",
            )
        )

        return items_deleted, bytes_freed

    def get_status(self) -> dict:
        """Get storage status."""
        total_bytes = sum(len(item.data) for item in self._items.values())

        return {
            "available": True,
            "total_items": len(self._items),
            "total_bytes": total_bytes,
            "ephemeral_handles": len(self._ephemeral_handles),
            "encryption_enabled": self._policy.encrypt_at_rest,
            "policy": {
                "default_retention_seconds": self._policy.default_retention_seconds,
                "secure_delete": self._policy.secure_delete,
            },
        }

    def get_policy(self) -> RetentionPolicy:
        """Get current retention policy."""
        return self._policy


def main() -> None:
    """Entry point for Storage service."""
    service = StorageService()
    service.run()


if __name__ == "__main__":
    main()


