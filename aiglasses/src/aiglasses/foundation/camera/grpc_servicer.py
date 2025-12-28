"""gRPC servicer for Camera service."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, AsyncIterator

import grpc
from grpc import aio as grpc_aio

if TYPE_CHECKING:
    from aiglasses.foundation.camera.service import CameraService


class CameraServicer:
    """gRPC servicer for Camera service."""

    def __init__(self, service: CameraService) -> None:
        self.service = service

    async def GetHealth(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get service health."""
        status = self.service.get_status()
        return {
            "service_name": "camera",
            "status": "healthy" if status["available"] else "unhealthy",
            "message": status.get("error_message", ""),
        }

    async def Snapshot(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Capture a snapshot."""
        format = request.get("format", "jpeg")
        quality = request.get("quality", 85)
        include_detections = request.get("include_detections", True)

        frame = await self.service.snapshot(format, quality, include_detections)

        return {
            "frame_ref": {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "width": frame.width,
                "height": frame.height,
                "format": frame.format,
            },
            "image_data": base64.b64encode(frame.data).decode("ascii"),
            "detections": frame.detections,
            "metadata": frame.metadata,
        }

    async def GetFrame(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get a cached frame."""
        frame_id = request.get("frame_id")
        if not frame_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "frame_id required")

        frame = await self.service.get_frame(frame_id)
        if not frame:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Frame {frame_id} not found")

        return {
            "frame_ref": {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "width": frame.width,
                "height": frame.height,
                "format": frame.format,
            },
            "data": base64.b64encode(frame.data).decode("ascii"),
            "format": frame.format,
            "metadata": frame.metadata,
        }

    async def StreamFrames(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[dict]:
        """Stream frames."""
        fps = request.get("fps", 10)
        format = request.get("format", "jpeg")
        quality = request.get("quality", 70)
        max_frames = request.get("max_frames", 0)
        include_detections = request.get("include_detections", False)

        async for frame in self.service.stream_frames(
            fps, format, quality, include_detections, max_frames
        ):
            yield {
                "frame_ref": {
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp,
                    "width": frame.width,
                    "height": frame.height,
                    "format": frame.format,
                },
                "data": base64.b64encode(frame.data).decode("ascii"),
                "format": frame.format,
                "metadata": frame.metadata,
            }

    async def GetDetections(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get detections for a frame."""
        frame_id = request.get("frame_id")

        try:
            frame, detections = await self.service.get_detections(frame_id)
        except ValueError as e:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(e))

        return {
            "frame_ref": {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "width": frame.width,
                "height": frame.height,
                "format": frame.format,
            },
            "detections": detections,
            "inference_time_ms": frame.metadata.get("inference_time_ms", 0),
        }

    async def GetStatus(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get camera status."""
        return self.service.get_status()


def add_servicer(server: grpc_aio.Server, servicer: CameraServicer) -> None:
    """Add Camera servicer to gRPC server."""
    from aiglasses.common.grpc_utils import GenericServiceHandler

    handler = GenericServiceHandler(
        servicer,
        service_name="aiglasses.camera.CameraService",
        methods={
            "GetHealth": ("unary", "unary"),
            "Snapshot": ("unary", "unary"),
            "GetFrame": ("unary", "unary"),
            "StreamFrames": ("unary", "stream"),
            "GetDetections": ("unary", "unary"),
            "GetStatus": ("unary", "unary"),
        },
    )
    server.add_generic_rpc_handlers((handler,))


