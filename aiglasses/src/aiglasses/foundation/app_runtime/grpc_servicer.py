"""gRPC servicer for App Runtime service."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc
from grpc import aio as grpc_aio

if TYPE_CHECKING:
    from aiglasses.foundation.app_runtime.service import AppRuntimeService


class AppRuntimeServicer:
    """gRPC servicer for App Runtime service."""

    def __init__(self, service: AppRuntimeService) -> None:
        self.service = service

    async def GetHealth(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get service health."""
        return {
            "service_name": "app_runtime",
            "status": "healthy",
            "message": "",
        }

    async def GetStatus(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get runtime status."""
        return self.service.get_status()

    async def ListApps(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """List installed apps."""
        apps = self.service.list_apps()
        return {
            "apps": [
                {
                    "id": app.id,
                    "name": app.manifest.name,
                    "version": app.manifest.version,
                    "state": app.state.value,
                    "pid": app.pid,
                }
                for app in apps
            ]
        }

    async def GetApp(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get app info."""
        app_id = request.get("app_id", "")
        app = self.service.get_app(app_id)
        if not app:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"App {app_id} not found")

        return {
            "id": app.id,
            "name": app.manifest.name,
            "version": app.manifest.version,
            "description": app.manifest.description,
            "state": app.state.value,
            "granted_permissions": [p.value for p in app.granted_permissions],
            "pid": app.pid,
        }

    async def Start(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Start an app."""
        app_id = request.get("app_id", "")
        env_override = request.get("env_override", {})

        try:
            pid = await self.service.start_app(app_id, env_override)
            return {"success": True, "message": "App started", "pid": pid}
        except Exception as e:
            return {"success": False, "message": str(e), "pid": 0}

    async def Stop(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Stop an app."""
        app_id = request.get("app_id", "")
        force = request.get("force", False)
        timeout = request.get("timeout_seconds", 5)

        try:
            await self.service.stop_app(app_id, force, timeout)
            return {"success": True, "message": "App stopped"}
        except Exception as e:
            return {"success": False, "message": str(e)}


def add_servicer(server: grpc_aio.Server, servicer: AppRuntimeServicer) -> None:
    """Add App Runtime servicer to gRPC server."""
    from aiglasses.common.grpc_utils import GenericServiceHandler

    handler = GenericServiceHandler(
        servicer,
        service_name="aiglasses.app_runtime.AppRuntimeService",
        methods={
            "GetHealth": ("unary", "unary"),
            "GetStatus": ("unary", "unary"),
            "ListApps": ("unary", "unary"),
            "GetApp": ("unary", "unary"),
            "Start": ("unary", "unary"),
            "Stop": ("unary", "unary"),
        },
    )
    server.add_generic_rpc_handlers((handler,))


