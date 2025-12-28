"""gRPC servicer for Device Manager."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, AsyncIterator

import grpc
from grpc import aio as grpc_aio

if TYPE_CHECKING:
    from aiglasses.foundation.device_manager.service import DeviceManagerService


class DeviceManagerServicer:
    """gRPC servicer for Device Manager.

    This is a simplified implementation that doesn't require proto compilation.
    In production, this would use generated protobuf classes.
    """

    def __init__(self, service: DeviceManagerService) -> None:
        self.service = service

    async def GetHealth(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get system health."""
        return await self.service.get_system_health()

    async def GetServiceStatus(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get status of a specific service."""
        service_name = request.get("service_name", "")
        status = await self.service.get_service_status(service_name)
        if status is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Service {service_name} not found")
        return status

    async def ListServices(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """List all services."""
        health = await self.service.get_system_health()
        services = []
        for name, status in health.get("services", {}).items():
            services.append({
                "name": name,
                "health": status,
                "port": getattr(self.service.config.ports, name, None),
            })
        return {"services": services}

    async def GetConfig(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get system configuration."""
        return await self.service.get_config()

    async def GetMetrics(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get system metrics."""
        health = await self.service.get_system_health()
        return {
            "timestamp": time.time(),
            "cpu_usage": health["resources"]["cpu_percent"],
            "memory_usage": health["resources"]["memory_percent"],
            "disk_usage": health["resources"]["disk_percent"],
            "temperature": health["resources"]["temperature_celsius"],
            "uptime_seconds": health["uptime_seconds"],
        }

    async def SubscribeEvents(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[dict]:
        """Subscribe to system events."""
        event_types = request.get("event_types", [])
        async for event in self.service.subscribe_events(event_types):
            yield {
                "event_type": event.topic,
                "source": event.source,
                "timestamp": event.timestamp,
                "data": event.data,
            }


def add_servicer(server: grpc_aio.Server, servicer: DeviceManagerServicer) -> None:
    """Add Device Manager servicer to gRPC server.

    This is a simplified registration that creates a generic service handler.
    In production, this would use generated add_*_to_server functions.
    """
    from aiglasses.common.grpc_utils import GenericServiceHandler

    handler = GenericServiceHandler(
        servicer,
        service_name="aiglasses.device_manager.DeviceManagerService",
        methods={
            "GetHealth": ("unary", "unary"),
            "GetServiceStatus": ("unary", "unary"),
            "ListServices": ("unary", "unary"),
            "GetConfig": ("unary", "unary"),
            "GetMetrics": ("unary", "unary"),
            "SubscribeEvents": ("unary", "stream"),
        },
    )
    server.add_generic_rpc_handlers((handler,))


