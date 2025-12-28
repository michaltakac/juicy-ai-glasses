"""Camera service implementation."""

from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from grpc import aio as grpc_aio

from aiglasses.common import BaseService, get_logger
from aiglasses.common.events import Event, get_event_bus
from aiglasses.config import Config


@dataclass
class Frame:
    """Captured frame data."""

    frame_id: str
    data: bytes
    width: int
    height: int
    format: str
    timestamp: float
    detections: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Detection:
    """Object detection result."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    model_source: str = "imx500"
    attributes: dict = field(default_factory=dict)


class CameraBackend:
    """Abstract camera backend."""

    async def setup(self) -> None:
        """Setup camera."""
        pass

    async def teardown(self) -> None:
        """Teardown camera."""
        pass

    async def capture(
        self,
        format: str = "jpeg",
        quality: int = 85,
    ) -> Frame:
        """Capture a frame."""
        raise NotImplementedError

    async def get_detections(self, frame: Frame) -> list[Detection]:
        """Get detections for a frame."""
        raise NotImplementedError

    def get_status(self) -> dict:
        """Get camera status."""
        raise NotImplementedError


class MockCameraBackend(CameraBackend):
    """Mock camera backend for testing."""

    def __init__(self) -> None:
        self._available = True
        self._frame_count = 0

    async def setup(self) -> None:
        """Setup mock camera."""
        pass

    async def teardown(self) -> None:
        """Teardown mock camera."""
        pass

    async def capture(
        self,
        format: str = "jpeg",
        quality: int = 85,
    ) -> Frame:
        """Capture a mock frame."""
        self._frame_count += 1

        # Generate a simple test image
        from PIL import Image

        img = Image.new("RGB", (640, 480), color=(73, 109, 137))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        image_data = buffer.getvalue()

        return Frame(
            frame_id=str(uuid.uuid4()),
            data=image_data,
            width=640,
            height=480,
            format=format,
            timestamp=time.time(),
            metadata={
                "frame_number": self._frame_count,
                "exposure": 0.01,
                "gain": 1.0,
            },
        )

    async def get_detections(self, frame: Frame) -> list[Detection]:
        """Get mock detections."""
        # Return mock detections
        return [
            Detection(
                label="person",
                confidence=0.92,
                bbox=(0.1, 0.2, 0.5, 0.8),
                model_source="mock",
            ),
            Detection(
                label="laptop",
                confidence=0.85,
                bbox=(0.6, 0.3, 0.9, 0.7),
                model_source="mock",
            ),
        ]

    def get_status(self) -> dict:
        """Get mock camera status."""
        return {
            "available": self._available,
            "state": "idle",
            "capabilities": {
                "supported_resolutions": [[640, 480], [1920, 1080]],
                "max_fps": 30,
                "supported_formats": ["jpeg", "png"],
                "has_imx500": False,
                "available_models": [],
            },
        }


class PiCameraBackend(CameraBackend):
    """Raspberry Pi camera backend using picamera2."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._camera = None
        self._available = False
        self.logger = get_logger("pi_camera_backend")

    async def setup(self) -> None:
        """Setup Pi camera."""
        try:
            from picamera2 import Picamera2

            self._camera = Picamera2()

            # Configure camera
            config = self._camera.create_still_configuration(
                main={"size": self.config.camera.resolution}
            )
            self._camera.configure(config)
            self._camera.start()

            self._available = True
            self.logger.info("pi_camera_initialized")

        except ImportError:
            self.logger.warning("picamera2_not_available")
            self._available = False
        except Exception as e:
            self.logger.exception("pi_camera_setup_failed", error=str(e))
            self._available = False

    async def teardown(self) -> None:
        """Teardown Pi camera."""
        if self._camera:
            self._camera.stop()
            self._camera.close()
            self._camera = None

    async def capture(
        self,
        format: str = "jpeg",
        quality: int = 85,
    ) -> Frame:
        """Capture a frame from Pi camera."""
        if not self._camera:
            raise RuntimeError("Camera not initialized")

        # Capture frame
        array = self._camera.capture_array()

        # Convert to image format
        from PIL import Image

        img = Image.fromarray(array)
        buffer = io.BytesIO()

        if format == "jpeg":
            img.save(buffer, format="JPEG", quality=quality)
        elif format == "png":
            img.save(buffer, format="PNG")
        else:
            img.save(buffer, format="JPEG", quality=quality)

        image_data = buffer.getvalue()

        return Frame(
            frame_id=str(uuid.uuid4()),
            data=image_data,
            width=img.width,
            height=img.height,
            format=format,
            timestamp=time.time(),
            metadata={
                "exposure": self._camera.capture_metadata().get("ExposureTime", 0),
                "gain": self._camera.capture_metadata().get("AnalogueGain", 1.0),
            },
        )

    async def get_detections(self, frame: Frame) -> list[Detection]:
        """Get IMX500 detections.

        Note: Full IMX500 integration requires the Sony IMX500 SDK.
        This is a placeholder that returns empty detections.
        """
        # TODO: Integrate with IMX500 SDK for on-sensor inference
        return []

    def get_status(self) -> dict:
        """Get Pi camera status."""
        return {
            "available": self._available,
            "state": "idle" if self._available else "error",
            "capabilities": {
                "supported_resolutions": [[640, 480], [1920, 1080], [4608, 2592]],
                "max_fps": 30,
                "supported_formats": ["jpeg", "png", "raw"],
                "has_imx500": self.config.camera.imx500_enabled,
                "available_models": ["mobilenet_v2"] if self.config.camera.imx500_enabled else [],
            },
        }


class CameraService(BaseService):
    """Camera service.

    Responsibilities:
    - Frame capture from IMX500 camera
    - On-sensor inference results
    - Frame storage and retrieval
    """

    def __init__(self, config: Config | None = None, mock_mode: bool = False) -> None:
        super().__init__("camera", config, mock_mode)
        self._backend: CameraBackend | None = None
        self._frame_cache: dict[str, Frame] = {}
        self._cache_max_size = 100
        self._event_bus = get_event_bus()

    @property
    def port(self) -> int:
        return self.config.ports.camera

    async def setup(self) -> None:
        """Setup Camera service."""
        self.logger.info("camera_service_setup", mock_mode=self.mock_mode)

        if self.mock_mode:
            self._backend = MockCameraBackend()
        else:
            self._backend = PiCameraBackend(self.config)

        await self._backend.setup()

    async def teardown(self) -> None:
        """Teardown Camera service."""
        self.logger.info("camera_service_teardown")

        if self._backend:
            await self._backend.teardown()

        self._frame_cache.clear()

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services."""
        from aiglasses.foundation.camera.grpc_servicer import CameraServicer, add_servicer

        servicer = CameraServicer(self)
        add_servicer(server, servicer)

    async def snapshot(
        self,
        format: str = "jpeg",
        quality: int = 85,
        include_detections: bool = True,
    ) -> Frame:
        """Capture a snapshot.

        Args:
            format: Image format (jpeg, png).
            quality: JPEG quality (1-100).
            include_detections: Run detection on the frame.

        Returns:
            Captured frame with optional detections.
        """
        if not self._backend:
            raise RuntimeError("Camera not initialized")

        start_time = time.time()

        # Capture frame
        frame = await self._backend.capture(format, quality)

        # Run detections if requested
        if include_detections:
            detections = await self._backend.get_detections(frame)
            frame.detections = [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": {"x_min": d.bbox[0], "y_min": d.bbox[1], "x_max": d.bbox[2], "y_max": d.bbox[3]},
                    "model_source": d.model_source,
                }
                for d in detections
            ]

        # Update metadata
        frame.metadata["capture_time_ms"] = int((time.time() - start_time) * 1000)

        # Cache the frame
        self._cache_frame(frame)

        # Publish event
        await self._event_bus.publish(
            Event(
                topic="camera.snapshot",
                data={
                    "frame_id": frame.frame_id,
                    "width": frame.width,
                    "height": frame.height,
                    "detection_count": len(frame.detections),
                },
                source="camera",
            )
        )

        return frame

    def _cache_frame(self, frame: Frame) -> None:
        """Cache a frame for later retrieval."""
        # Evict oldest frames if cache is full
        while len(self._frame_cache) >= self._cache_max_size:
            oldest_id = min(self._frame_cache.keys(), key=lambda k: self._frame_cache[k].timestamp)
            del self._frame_cache[oldest_id]

        self._frame_cache[frame.frame_id] = frame

    async def get_frame(self, frame_id: str) -> Frame | None:
        """Get a cached frame by ID.

        Args:
            frame_id: Frame ID.

        Returns:
            Frame or None if not found.
        """
        return self._frame_cache.get(frame_id)

    async def get_detections(self, frame_id: str | None = None) -> tuple[Frame, list[dict]]:
        """Get detections for a frame.

        Args:
            frame_id: Frame ID. If None, captures a new frame.

        Returns:
            Tuple of (frame, detections).
        """
        if frame_id:
            frame = self._frame_cache.get(frame_id)
            if not frame:
                raise ValueError(f"Frame {frame_id} not found")
        else:
            frame = await self.snapshot(include_detections=True)

        return frame, frame.detections

    async def stream_frames(
        self,
        fps: int = 10,
        format: str = "jpeg",
        quality: int = 70,
        include_detections: bool = False,
        max_frames: int = 0,
    ) -> AsyncIterator[Frame]:
        """Stream frames continuously.

        Args:
            fps: Target frames per second.
            format: Image format.
            quality: JPEG quality.
            include_detections: Include detections in each frame.
            max_frames: Maximum frames to stream (0 = unlimited).

        Yields:
            Frames.
        """
        interval = 1.0 / fps
        frame_count = 0

        while max_frames == 0 or frame_count < max_frames:
            start_time = time.time()

            frame = await self.snapshot(format, quality, include_detections)
            yield frame

            frame_count += 1

            # Maintain target FPS
            elapsed = time.time() - start_time
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

    def get_status(self) -> dict:
        """Get camera status.

        Returns:
            Camera status.
        """
        if not self._backend:
            return {
                "available": False,
                "state": "not_initialized",
                "error_message": "Camera not initialized",
            }

        status = self._backend.get_status()
        status["cached_frames"] = len(self._frame_cache)
        return status


def main() -> None:
    """Entry point for Camera service."""
    service = CameraService()
    service.run()


if __name__ == "__main__":
    main()


