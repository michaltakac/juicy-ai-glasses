"""gRPC servicer for Storage service."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import grpc
from grpc import aio as grpc_aio

if TYPE_CHECKING:
    from aiglasses.foundation.storage.service import StorageService


class StorageServicer:
    """gRPC servicer for Storage service."""

    def __init__(self, service: StorageService) -> None:
        self.service = service

    async def GetHealth(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get service health."""
        return {
            "service_name": "storage",
            "status": "healthy",
            "message": "",
        }

    async def GetStatus(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get storage status."""
        return self.service.get_status()

    async def Store(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Store data."""
        key = request.get("key", "")
        data = base64.b64decode(request.get("data", ""))
        content_type = request.get("content_type", "application/octet-stream")
        category = request.get("options", {}).get("category", "other")
        app_id = request.get("options", {}).get("app_id")
        retention = request.get("options", {}).get("retention_seconds")
        metadata = request.get("metadata", {})

        item = await self.service.store(
            key, data, content_type, category, app_id, retention, metadata
        )

        return {
            "success": True,
            "key": item.key,
            "expires_at": item.expires_at,
        }

    async def Retrieve(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Retrieve data."""
        key = request.get("key", "")
        item = await self.service.retrieve(key)

        if not item:
            return {"found": False}

        return {
            "found": True,
            "data": base64.b64encode(item.data).decode("ascii"),
            "content_type": item.content_type,
            "metadata": item.metadata,
            "created_at": item.created_at,
            "expires_at": item.expires_at,
        }

    async def Delete(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Delete data."""
        key = request.get("key", "")
        secure = request.get("secure_delete", False)
        success = await self.service.delete(key, secure)
        return {"success": success, "message": "Deleted" if success else "Not found"}

    async def Purge(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Purge all data."""
        if not request.get("confirm", False):
            return {"success": False, "message": "Confirmation required"}

        category = request.get("category")
        app_id = request.get("app_id")
        items_deleted, bytes_freed = await self.service.purge(category, app_id)

        return {
            "success": True,
            "items_deleted": items_deleted,
            "bytes_freed": bytes_freed,
        }

    async def GetPolicy(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get retention policy."""
        policy = self.service.get_policy()
        return {
            "default_retention_seconds": policy.default_retention_seconds,
            "audio_retention_seconds": policy.audio_retention_seconds,
            "video_retention_seconds": policy.video_retention_seconds,
            "frame_retention_seconds": policy.frame_retention_seconds,
            "encrypt_at_rest": policy.encrypt_at_rest,
            "secure_delete": policy.secure_delete,
        }


def add_servicer(server: grpc_aio.Server, servicer: StorageServicer) -> None:
    """Add Storage servicer to gRPC server."""
    from aiglasses.common.grpc_utils import GenericServiceHandler

    handler = GenericServiceHandler(
        servicer,
        service_name="aiglasses.storage.StorageService",
        methods={
            "GetHealth": ("unary", "unary"),
            "GetStatus": ("unary", "unary"),
            "Store": ("unary", "unary"),
            "Retrieve": ("unary", "unary"),
            "Delete": ("unary", "unary"),
            "Purge": ("unary", "unary"),
            "GetPolicy": ("unary", "unary"),
        },
    )
    server.add_generic_rpc_handlers((handler,))


