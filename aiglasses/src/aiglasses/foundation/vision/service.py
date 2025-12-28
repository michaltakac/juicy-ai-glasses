"""Vision service implementation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from grpc import aio as grpc_aio

from aiglasses.common import BaseService, get_logger
from aiglasses.common.events import Event, get_event_bus
from aiglasses.config import Config


@dataclass
class Detection:
    """Object detection result."""

    label: str
    confidence: float
    bbox: dict  # {x_min, y_min, x_max, y_max}
    model_source: str
    attributes: dict = field(default_factory=dict)


@dataclass
class DetectResult:
    """Detection results for a frame."""

    frame_id: str
    detections: list[Detection]
    inference_time_ms: int
    model_used: str
    backend_times_ms: dict = field(default_factory=dict)


@dataclass
class DescribeResult:
    """Scene description result."""

    description: str
    frame_id: str
    detections: list[Detection]
    latency_ms: int


class VisionBackend:
    """Abstract vision backend."""

    async def setup(self) -> None:
        """Setup vision backend."""
        pass

    async def teardown(self) -> None:
        """Teardown vision backend."""
        pass

    async def detect(
        self,
        image_data: bytes,
        model: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[Detection]:
        """Detect objects in image."""
        raise NotImplementedError

    async def describe(
        self,
        image_data: bytes,
        detections: list[Detection],
        prompt: str | None = None,
    ) -> str:
        """Describe the scene."""
        raise NotImplementedError

    def get_status(self) -> dict:
        """Get backend status."""
        raise NotImplementedError


class MockVisionBackend(VisionBackend):
    """Mock vision backend for testing."""

    def __init__(self) -> None:
        self._available = True
        self._frame_count = 0

    async def setup(self) -> None:
        """Setup mock vision."""
        pass

    async def teardown(self) -> None:
        """Teardown mock vision."""
        pass

    async def detect(
        self,
        image_data: bytes,
        model: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[Detection]:
        """Mock detection."""
        self._frame_count += 1
        await asyncio.sleep(0.02)  # Simulate inference time

        return [
            Detection(
                label="person",
                confidence=0.92,
                bbox={"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8},
                model_source="mock",
            ),
            Detection(
                label="laptop",
                confidence=0.85,
                bbox={"x_min": 0.6, "y_min": 0.3, "x_max": 0.9, "y_max": 0.7},
                model_source="mock",
            ),
            Detection(
                label="cup",
                confidence=0.78,
                bbox={"x_min": 0.55, "y_min": 0.5, "x_max": 0.65, "y_max": 0.65},
                model_source="mock",
            ),
        ]

    async def describe(
        self,
        image_data: bytes,
        detections: list[Detection],
        prompt: str | None = None,
    ) -> str:
        """Mock scene description."""
        await asyncio.sleep(0.05)  # Simulate processing time

        if detections:
            labels = [d.label for d in detections]
            return f"I can see {', '.join(labels)} in the image."
        else:
            return "I can see a scene but cannot identify specific objects."

    def get_status(self) -> dict:
        """Get mock status."""
        return {
            "available": self._available,
            "imx500_available": False,
            "hailo_available": False,
            "loaded_models": [
                {"id": "mock", "name": "Mock Model", "backend": "mock", "ready": True}
            ],
            "pipeline_state": "idle",
            "metrics": {
                "avg_detection_ms": 20.0,
                "avg_describe_ms": 50.0,
                "total_frames_processed": self._frame_count,
            },
        }


class HailoVisionBackend(VisionBackend):
    """Vision backend using Hailo AI HAT+."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._available = False
        self._hailo_available = False
        self._frame_count = 0
        self.logger = get_logger("hailo_vision_backend")

    async def setup(self) -> None:
        """Setup Hailo backend."""
        if not self.config.vision.hailo_enabled:
            self.logger.info("hailo_disabled_in_config")
            return

        try:
            # Try to import HailoRT
            import hailo_platform
            self._hailo_available = True
            self._available = True
            self.logger.info("hailo_initialized")
        except ImportError:
            self.logger.warning("hailo_not_available")
            self._hailo_available = False

    async def teardown(self) -> None:
        """Teardown Hailo backend."""
        pass

    async def detect(
        self,
        image_data: bytes,
        model: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[Detection]:
        """Detect using Hailo."""
        if not self._hailo_available:
            return []

        self._frame_count += 1

        # TODO: Implement actual Hailo inference
        # This is a placeholder that returns mock results
        return [
            Detection(
                label="object",
                confidence=0.7,
                bbox={"x_min": 0.2, "y_min": 0.2, "x_max": 0.8, "y_max": 0.8},
                model_source="hailo",
            )
        ]

    async def describe(
        self,
        image_data: bytes,
        detections: list[Detection],
        prompt: str | None = None,
    ) -> str:
        """Describe scene based on detections."""
        if detections:
            labels = [d.label for d in detections]
            return f"Detected: {', '.join(labels)}"
        return "No objects detected"

    def get_status(self) -> dict:
        """Get Hailo status."""
        return {
            "available": self._available,
            "imx500_available": False,
            "hailo_available": self._hailo_available,
            "loaded_models": [],
            "pipeline_state": "idle" if self._available else "error",
            "metrics": {
                "avg_detection_ms": 0.0,
                "total_frames_processed": self._frame_count,
            },
        }


class VisionService(BaseService):
    """Vision service.

    Responsibilities:
    - Object detection pipeline
    - Scene description
    - IMX500 + Hailo fusion
    """

    def __init__(self, config: Config | None = None, mock_mode: bool = False) -> None:
        super().__init__("vision", config, mock_mode)
        self._backend: VisionBackend | None = None
        self._event_bus = get_event_bus()

    @property
    def port(self) -> int:
        return self.config.ports.vision

    async def setup(self) -> None:
        """Setup Vision service."""
        self.logger.info("vision_service_setup", mock_mode=self.mock_mode)

        if self.mock_mode:
            self._backend = MockVisionBackend()
        else:
            self._backend = HailoVisionBackend(self.config)

        await self._backend.setup()

    async def teardown(self) -> None:
        """Teardown Vision service."""
        self.logger.info("vision_service_teardown")

        if self._backend:
            await self._backend.teardown()

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services."""
        from aiglasses.foundation.vision.grpc_servicer import VisionServicer, add_servicer

        servicer = VisionServicer(self)
        add_servicer(server, servicer)

    async def detect_objects(
        self,
        image_data: bytes,
        frame_id: str = "",
        model: str | None = None,
        min_confidence: float | None = None,
    ) -> DetectResult:
        """Detect objects in an image.

        Args:
            image_data: Image bytes.
            frame_id: Frame identifier.
            model: Model to use.
            min_confidence: Minimum confidence threshold.

        Returns:
            Detection results.
        """
        if not self._backend:
            raise RuntimeError("Vision backend not initialized")

        start_time = time.time()
        min_confidence = min_confidence or self.config.vision.default_confidence

        detections = await self._backend.detect(image_data, model, min_confidence)
        inference_time_ms = int((time.time() - start_time) * 1000)

        # Filter by confidence
        detections = [d for d in detections if d.confidence >= min_confidence]

        result = DetectResult(
            frame_id=frame_id,
            detections=detections,
            inference_time_ms=inference_time_ms,
            model_used=model or self.config.vision.default_model,
        )

        # Publish event
        await self._event_bus.publish(
            Event(
                topic="vision.detections",
                data={
                    "frame_id": frame_id,
                    "detection_count": len(detections),
                    "labels": [d.label for d in detections],
                },
                source="vision",
            )
        )

        return result

    async def describe_scene(
        self,
        image_data: bytes,
        frame_id: str = "",
        detections: list[Detection] | None = None,
        prompt: str | None = None,
    ) -> DescribeResult:
        """Describe a scene.

        Args:
            image_data: Image bytes.
            frame_id: Frame identifier.
            detections: Pre-computed detections.
            prompt: Optional context prompt.

        Returns:
            Scene description.
        """
        if not self._backend:
            raise RuntimeError("Vision backend not initialized")

        start_time = time.time()

        # Run detection if not provided
        if detections is None:
            detect_result = await self.detect_objects(image_data, frame_id)
            detections = detect_result.detections

        description = await self._backend.describe(image_data, detections, prompt)
        latency_ms = int((time.time() - start_time) * 1000)

        return DescribeResult(
            description=description,
            frame_id=frame_id,
            detections=detections,
            latency_ms=latency_ms,
        )

    def get_status(self) -> dict:
        """Get vision status."""
        if not self._backend:
            return {"available": False, "error_message": "Backend not initialized"}
        return self._backend.get_status()


def main() -> None:
    """Entry point for Vision service."""
    service = VisionService()
    service.run()


if __name__ == "__main__":
    main()


