"""gRPC servicer for Audio service."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, AsyncIterator

import grpc
from grpc import aio as grpc_aio

if TYPE_CHECKING:
    from aiglasses.foundation.audio.service import AudioService


class AudioServicer:
    """gRPC servicer for Audio service."""

    def __init__(self, service: AudioService) -> None:
        self.service = service

    async def GetHealth(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get service health."""
        status = self.service.get_status()
        return {
            "service_name": "audio",
            "status": "healthy" if status.get("available") else "unhealthy",
            "message": status.get("error_message", ""),
        }

    async def GetStatus(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get audio status."""
        return self.service.get_status()

    async def ListDevices(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """List audio devices."""
        devices = await self.service.list_devices()
        return {
            "devices": [
                {
                    "address": d.address,
                    "name": d.name,
                    "type": d.type,
                    "connected": d.connected,
                    "paired": d.paired,
                    "supported_profiles": d.supported_profiles,
                    "battery_level": d.battery_level,
                    "signal_strength": d.signal_strength,
                }
                for d in devices
            ]
        }

    async def Connect(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Connect to a device."""
        address = request.get("address", "")
        profile = request.get("profile", "auto")

        device = await self.service.connect(address, profile)
        if device:
            return {
                "success": True,
                "message": f"Connected to {device.name}",
                "device": {
                    "address": device.address,
                    "name": device.name,
                    "type": device.type,
                    "connected": device.connected,
                },
            }
        else:
            return {
                "success": False,
                "message": f"Failed to connect to {address}",
            }

    async def Disconnect(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Disconnect from current device."""
        await self.service.disconnect()
        return {"success": True, "message": "Disconnected"}

    async def OpenMic(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[dict]:
        """Open microphone stream."""
        sample_rate = request.get("sample_rate")
        channels = request.get("channels")
        chunk_size_ms = request.get("chunk_size_ms")

        async for chunk in self.service.open_mic(sample_rate, channels, chunk_size_ms):
            yield {
                "data": base64.b64encode(chunk.data).decode("ascii"),
                "timestamp_ms": chunk.timestamp_ms,
                "sample_rate": chunk.sample_rate,
                "channels": chunk.channels,
                "format": chunk.format,
                "sequence": chunk.sequence,
            }

    async def PlayBytes(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Play audio bytes."""
        audio_data = base64.b64decode(request.get("audio_data", ""))
        format = request.get("format", "wav")
        sample_rate = request.get("sample_rate", 22050)

        await self.service.play_bytes(audio_data, format, sample_rate)
        return {"success": True, "message": "Playback complete"}


def add_servicer(server: grpc_aio.Server, servicer: AudioServicer) -> None:
    """Add Audio servicer to gRPC server."""
    from aiglasses.common.grpc_utils import GenericServiceHandler

    handler = GenericServiceHandler(
        servicer,
        service_name="aiglasses.audio.AudioService",
        methods={
            "GetHealth": ("unary", "unary"),
            "GetStatus": ("unary", "unary"),
            "ListDevices": ("unary", "unary"),
            "Connect": ("unary", "unary"),
            "Disconnect": ("unary", "unary"),
            "OpenMic": ("unary", "stream"),
            "PlayBytes": ("unary", "unary"),
        },
    )
    server.add_generic_rpc_handlers((handler,))


