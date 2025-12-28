"""gRPC servicer for Vision service."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import grpc
from grpc import aio as grpc_aio

if TYPE_CHECKING:
    from aiglasses.foundation.vision.service import VisionService


class VisionServicer:
    """gRPC servicer for Vision service."""

    def __init__(self, service: VisionService) -> None:
        self.service = service

    async def GetHealth(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get service health."""
        status = self.service.get_status()
        return {
            "service_name": "vision",
            "status": "healthy" if status.get("available") else "unhealthy",
            "message": status.get("error_message", ""),
        }

    async def GetStatus(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get vision status."""
        return self.service.get_status()

    async def DetectObjects(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Detect objects in an image."""
        frame_id = request.get("frame_id", "")
        image_data = base64.b64decode(request.get("image_data", ""))
        model = request.get("model")
        min_confidence = request.get("min_confidence", 0.5)

        result = await self.service.detect_objects(
            image_data, frame_id, model, min_confidence
        )

        return {
            "frame_ref": {"frame_id": result.frame_id},
            "detections": [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "model_source": d.model_source,
                    "attributes": d.attributes,
                }
                for d in result.detections
            ],
            "inference_time_ms": result.inference_time_ms,
            "model_used": result.model_used,
        }

    async def DescribeScene(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Describe a scene."""
        frame_id = request.get("frame_id", "")
        image_data = base64.b64decode(request.get("image_data", ""))
        prompt = request.get("prompt")

        result = await self.service.describe_scene(image_data, frame_id, prompt=prompt)

        return {
            "description": result.description,
            "frame_ref": {"frame_id": result.frame_id},
            "detections": [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "model_source": d.model_source,
                }
                for d in result.detections
            ],
            "latency_ms": result.latency_ms,
        }


def add_servicer(server: grpc_aio.Server, servicer: VisionServicer) -> None:
    """Add Vision servicer to gRPC server."""
    from aiglasses.common.grpc_utils import GenericServiceHandler

    handler = GenericServiceHandler(
        servicer,
        service_name="aiglasses.vision.VisionService",
        methods={
            "GetHealth": ("unary", "unary"),
            "GetStatus": ("unary", "unary"),
            "DetectObjects": ("unary", "unary"),
            "DescribeScene": ("unary", "unary"),
        },
    )
    server.add_generic_rpc_handlers((handler,))


