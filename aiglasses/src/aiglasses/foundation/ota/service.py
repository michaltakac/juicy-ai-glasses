"""OTA service implementation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from grpc import aio as grpc_aio

from aiglasses.common import BaseService, get_logger
from aiglasses.common.events import Event, get_event_bus
from aiglasses.config import Config
from aiglasses import __version__


class UpdateState(Enum):
    """Update state enum."""

    IDLE = "idle"
    CHECKING = "checking"
    DOWNLOADING = "downloading"
    APPLYING = "applying"
    ERROR = "error"


@dataclass
class UpdateInfo:
    """Update information."""

    id: str
    type: str  # "system", "foundation", "model", "app"
    name: str
    current_version: str
    new_version: str
    description: str = ""
    size_bytes: int = 0
    release_notes: str = ""
    requires_reboot: bool = False
    channel: str = "stable"
    released_at: float = 0.0


@dataclass
class UpdateRecord:
    """Update history record."""

    id: str
    type: str
    name: str
    from_version: str
    to_version: str
    status: str  # "success", "failed", "rolled_back"
    applied_at: float
    error: str = ""


class OTAService(BaseService):
    """OTA service.

    Responsibilities:
    - Check for updates
    - Download updates
    - Apply updates
    - Rollback support
    """

    def __init__(self, config: Config | None = None, mock_mode: bool = False) -> None:
        super().__init__("ota", config, mock_mode)
        self._state = UpdateState.IDLE
        self._pending_update: UpdateInfo | None = None
        self._download_progress = 0.0
        self._history: list[UpdateRecord] = []
        self._event_bus = get_event_bus()
        self._last_check: float | None = None

    @property
    def port(self) -> int:
        return self.config.ports.ota

    async def setup(self) -> None:
        """Setup OTA service."""
        self.logger.info("ota_service_setup", mock_mode=self.mock_mode)

    async def teardown(self) -> None:
        """Teardown OTA service."""
        self.logger.info("ota_service_teardown")

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services."""
        from aiglasses.foundation.ota.grpc_servicer import OTAServicer, add_servicer

        servicer = OTAServicer(self)
        add_servicer(server, servicer)

    async def check_updates(
        self,
        channel: str | None = None,
        include_apps: bool = True,
    ) -> tuple[list[UpdateInfo], list[UpdateInfo]]:
        """Check for available updates.

        Args:
            channel: Update channel.
            include_apps: Include app updates.

        Returns:
            Tuple of (system_updates, app_updates).
        """
        self._state = UpdateState.CHECKING
        channel = channel or self.config.ota.channel

        try:
            # In real implementation, this would query the update server
            # For now, return empty lists
            system_updates: list[UpdateInfo] = []
            app_updates: list[UpdateInfo] = []

            self._last_check = time.time()
            self._state = UpdateState.IDLE

            return system_updates, app_updates

        except Exception as e:
            self._state = UpdateState.ERROR
            self.logger.exception("check_updates_failed", error=str(e))
            raise

    async def download_update(
        self,
        update_id: str,
    ) -> AsyncIterator[dict]:
        """Download an update.

        Args:
            update_id: Update ID.

        Yields:
            Download progress updates.
        """
        self._state = UpdateState.DOWNLOADING
        self._download_progress = 0.0

        try:
            # Simulate download progress
            for progress in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                await asyncio.sleep(0.5)
                self._download_progress = progress
                yield {
                    "update_id": update_id,
                    "progress": progress,
                    "state": "downloading" if progress < 1.0 else "complete",
                }

            self._state = UpdateState.IDLE

        except Exception as e:
            self._state = UpdateState.ERROR
            yield {
                "update_id": update_id,
                "progress": self._download_progress,
                "state": "error",
                "error": str(e),
            }

    async def apply_update(
        self,
        update_id: str,
        auto_reboot: bool = False,
    ) -> AsyncIterator[dict]:
        """Apply an update.

        Args:
            update_id: Update ID.
            auto_reboot: Reboot automatically if required.

        Yields:
            Apply progress updates.
        """
        self._state = UpdateState.APPLYING

        try:
            # Simulate apply progress
            steps = ["preparing", "applying", "verifying", "complete"]
            for i, step in enumerate(steps):
                await asyncio.sleep(0.5)
                yield {
                    "update_id": update_id,
                    "state": step,
                    "progress": (i + 1) / len(steps),
                    "current_step": step,
                }

            # Record in history
            self._history.append(
                UpdateRecord(
                    id=update_id,
                    type="system",
                    name="Update",
                    from_version=__version__,
                    to_version="0.1.1",
                    status="success",
                    applied_at=time.time(),
                )
            )

            self._state = UpdateState.IDLE

        except Exception as e:
            self._state = UpdateState.ERROR
            self._history.append(
                UpdateRecord(
                    id=update_id,
                    type="system",
                    name="Update",
                    from_version=__version__,
                    to_version="0.1.1",
                    status="failed",
                    applied_at=time.time(),
                    error=str(e),
                )
            )
            yield {
                "update_id": update_id,
                "state": "error",
                "error": str(e),
            }

    async def rollback(self, component: str = "system") -> tuple[bool, str, str]:
        """Rollback to previous version.

        Args:
            component: Component to rollback.

        Returns:
            Tuple of (success, message, rolled_back_version).
        """
        # In real implementation, this would restore from backup
        return False, "Rollback not implemented in mock mode", ""

    def get_status(self) -> dict:
        """Get OTA status."""
        return {
            "available": True,
            "state": self._state.value,
            "pending_update": {
                "id": self._pending_update.id,
                "name": self._pending_update.name,
                "new_version": self._pending_update.new_version,
            }
            if self._pending_update
            else None,
            "download_progress": self._download_progress,
            "last_check": self._last_check,
        }

    def get_history(self) -> list[UpdateRecord]:
        """Get update history."""
        return list(self._history)

    def get_versions(self) -> dict:
        """Get current versions."""
        return {
            "system_version": __version__,
            "foundation_version": __version__,
            "service_versions": {
                "device_manager": __version__,
                "camera": __version__,
                "audio": __version__,
                "speech": __version__,
                "vision": __version__,
                "llm_gateway": __version__,
                "app_runtime": __version__,
                "storage": __version__,
                "ota": __version__,
            },
            "channel": self.config.ota.channel,
        }


def main() -> None:
    """Entry point for OTA service."""
    service = OTAService()
    service.run()


if __name__ == "__main__":
    main()


