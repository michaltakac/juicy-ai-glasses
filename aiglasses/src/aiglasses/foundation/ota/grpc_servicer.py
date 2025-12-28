"""gRPC servicer for OTA service."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

import grpc
from grpc import aio as grpc_aio

if TYPE_CHECKING:
    from aiglasses.foundation.ota.service import OTAService


class OTAServicer:
    """gRPC servicer for OTA service."""

    def __init__(self, service: OTAService) -> None:
        self.service = service

    async def GetHealth(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get service health."""
        return {
            "service_name": "ota",
            "status": "healthy",
            "message": "",
        }

    async def GetStatus(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get OTA status."""
        return self.service.get_status()

    async def CheckUpdates(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Check for updates."""
        channel = request.get("channel")
        include_apps = request.get("include_apps", True)

        system_updates, app_updates = await self.service.check_updates(channel, include_apps)

        return {
            "updates_available": bool(system_updates or app_updates),
            "system_updates": [
                {
                    "id": u.id,
                    "type": u.type,
                    "name": u.name,
                    "current_version": u.current_version,
                    "new_version": u.new_version,
                    "size_bytes": u.size_bytes,
                }
                for u in system_updates
            ],
            "app_updates": [
                {
                    "id": u.id,
                    "type": u.type,
                    "name": u.name,
                    "current_version": u.current_version,
                    "new_version": u.new_version,
                    "size_bytes": u.size_bytes,
                }
                for u in app_updates
            ],
        }

    async def DownloadUpdate(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[dict]:
        """Download an update."""
        update_id = request.get("update_id", "")
        async for progress in self.service.download_update(update_id):
            yield progress

    async def ApplyUpdate(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[dict]:
        """Apply an update."""
        update_id = request.get("update_id", "")
        auto_reboot = request.get("auto_reboot", False)
        async for progress in self.service.apply_update(update_id, auto_reboot):
            yield progress

    async def Rollback(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Rollback to previous version."""
        component = request.get("component", "system")
        success, message, version = await self.service.rollback(component)
        return {
            "success": success,
            "message": message,
            "rolled_back_version": version,
        }

    async def GetHistory(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get update history."""
        history = self.service.get_history()
        return {
            "records": [
                {
                    "id": r.id,
                    "type": r.type,
                    "name": r.name,
                    "from_version": r.from_version,
                    "to_version": r.to_version,
                    "status": r.status,
                    "applied_at": r.applied_at,
                    "error": r.error,
                }
                for r in history
            ]
        }

    async def GetVersions(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get current versions."""
        return self.service.get_versions()


def add_servicer(server: grpc_aio.Server, servicer: OTAServicer) -> None:
    """Add OTA servicer to gRPC server."""
    from aiglasses.common.grpc_utils import GenericServiceHandler

    handler = GenericServiceHandler(
        servicer,
        service_name="aiglasses.ota.OTAService",
        methods={
            "GetHealth": ("unary", "unary"),
            "GetStatus": ("unary", "unary"),
            "CheckUpdates": ("unary", "unary"),
            "DownloadUpdate": ("unary", "stream"),
            "ApplyUpdate": ("unary", "stream"),
            "Rollback": ("unary", "unary"),
            "GetHistory": ("unary", "unary"),
            "GetVersions": ("unary", "unary"),
        },
    )
    server.add_generic_rpc_handlers((handler,))


