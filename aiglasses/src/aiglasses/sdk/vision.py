"""SDK Vision API - camera and object detection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from aiglasses.common.logging import get_logger
from aiglasses.common.grpc_utils import GrpcClient
from aiglasses.config import Config


@dataclass
class Frame:
    """Captured frame."""

    frame_id: str
    data: bytes
    width: int
    height: int
    format: str
    timestamp: float
    metadata: dict = field(default_factory=dict)


@dataclass
class BoundingBox:
    """Bounding box for detection."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class Detection:
    """Object detection result."""

    label: str
    confidence: float
    bbox: BoundingBox
    model_source: str = "imx500"
    attributes: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Detection results for a frame."""

    frame_id: str
    detections: list[Detection]
    inference_time_ms: int
    model_used: str


@dataclass
class SceneDescription:
    """Scene description result."""

    description: str
    frame_id: str
    detections: list[Detection]
    latency_ms: int


class VisionAPI:
    """Vision API for camera and object detection.

    Provides methods for:
    - Frame capture (screenshot)
    - Object detection
    - Scene description
    """

    def __init__(self, config: Config, mock_mode: bool = False) -> None:
        """Initialize Vision API.

        Args:
            config: Configuration.
            mock_mode: Run in mock mode.
        """
        self.config = config
        self.mock_mode = mock_mode
        self.logger = get_logger("sdk.vision")

        self._camera_client: GrpcClient | None = None
        self._vision_client: GrpcClient | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to camera and vision services."""
        if self._connected:
            return

        if not self.mock_mode:
            self._camera_client = GrpcClient("localhost", self.config.ports.camera)
            self._vision_client = GrpcClient("localhost", self.config.ports.vision)
            await self._camera_client.connect()
            await self._vision_client.connect()

        self._connected = True
        self.logger.debug("vision_api_connected")

    async def disconnect(self) -> None:
        """Disconnect from services."""
        if self._camera_client:
            await self._camera_client.close()
        if self._vision_client:
            await self._vision_client.close()
        self._connected = False

    async def snapshot(
        self,
        format: str = "jpeg",
        quality: int = 85,
    ) -> Frame:
        """Capture a single frame (screenshot).

        Args:
            format: Image format ("jpeg", "png").
            quality: JPEG quality (1-100).

        Returns:
            Captured frame.

        Example:
            frame = await app.vision.snapshot()
            print(f"Captured frame {frame.frame_id} at {frame.width}x{frame.height}")
        """
        if self.mock_mode:
            await asyncio.sleep(0.05)  # Simulate capture time
            import uuid
            import time

            # Generate a simple mock image
            from PIL import Image
            import io

            img = Image.new("RGB", (640, 480), color=(73, 109, 137))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)

            return Frame(
                frame_id=str(uuid.uuid4()),
                data=buffer.getvalue(),
                width=640,
                height=480,
                format=format,
                timestamp=time.time(),
                metadata={"mock": True},
            )

        import base64
        response = await self._camera_client.call(
            "aiglasses.camera.CameraService",
            "Snapshot",
            {
                "format": format,
                "quality": quality,
                "include_detections": False,
            },
        )

        frame_ref = response.get("frame_ref", {})
        return Frame(
            frame_id=frame_ref.get("frame_id", ""),
            data=base64.b64decode(response.get("image_data", "")),
            width=frame_ref.get("width", 0),
            height=frame_ref.get("height", 0),
            format=frame_ref.get("format", format),
            timestamp=frame_ref.get("timestamp", 0),
            metadata=response.get("metadata", {}),
        )

    async def detect_objects(
        self,
        frame: Frame | None = None,
        min_confidence: float | None = None,
        labels: list[str] | None = None,
    ) -> DetectionResult:
        """Detect objects in a frame.

        Args:
            frame: Frame to analyze. If None, captures a new frame.
            min_confidence: Minimum confidence threshold.
            labels: Filter to specific labels.

        Returns:
            Detection results.

        Example:
            frame = await app.vision.snapshot()
            result = await app.vision.detect_objects(frame)
            for det in result.detections:
                print(f"Found {det.label} ({det.confidence:.0%})")
        """
        min_confidence = min_confidence or self.config.vision.default_confidence

        # Capture frame if not provided
        if frame is None:
            frame = await self.snapshot()

        if self.mock_mode:
            await asyncio.sleep(0.02)  # Simulate inference
            return DetectionResult(
                frame_id=frame.frame_id,
                detections=[
                    Detection(
                        label="person",
                        confidence=0.92,
                        bbox=BoundingBox(0.1, 0.2, 0.5, 0.8),
                        model_source="mock",
                    ),
                    Detection(
                        label="laptop",
                        confidence=0.85,
                        bbox=BoundingBox(0.6, 0.3, 0.9, 0.7),
                        model_source="mock",
                    ),
                ],
                inference_time_ms=20,
                model_used="mock",
            )

        import base64
        response = await self._vision_client.call(
            "aiglasses.vision.VisionService",
            "DetectObjects",
            {
                "frame_id": frame.frame_id,
                "image_data": base64.b64encode(frame.data).decode("ascii"),
                "min_confidence": min_confidence,
            },
        )

        detections = []
        for d in response.get("detections", []):
            bbox = d.get("bbox", {})
            detections.append(
                Detection(
                    label=d.get("label", ""),
                    confidence=d.get("confidence", 0.0),
                    bbox=BoundingBox(
                        bbox.get("x_min", 0),
                        bbox.get("y_min", 0),
                        bbox.get("x_max", 0),
                        bbox.get("y_max", 0),
                    ),
                    model_source=d.get("model_source", ""),
                    attributes=d.get("attributes", {}),
                )
            )

        # Filter by labels if specified
        if labels:
            detections = [d for d in detections if d.label in labels]

        return DetectionResult(
            frame_id=frame.frame_id,
            detections=detections,
            inference_time_ms=response.get("inference_time_ms", 0),
            model_used=response.get("model_used", ""),
        )

    async def describe_scene(
        self,
        frame: Frame | None = None,
        detections: list[Detection] | None = None,
        prompt: str | None = None,
    ) -> SceneDescription:
        """Describe the scene in natural language.

        Args:
            frame: Frame to describe. If None, captures a new frame.
            detections: Pre-computed detections to use.
            prompt: Optional prompt/question to guide description.

        Returns:
            Scene description.

        Example:
            frame = await app.vision.snapshot()
            desc = await app.vision.describe_scene(frame)
            print(desc.description)
        """
        # Capture frame if not provided
        if frame is None:
            frame = await self.snapshot()

        # Get detections if not provided
        if detections is None:
            result = await self.detect_objects(frame)
            detections = result.detections

        if self.mock_mode:
            await asyncio.sleep(0.05)
            labels = [d.label for d in detections]
            description = f"I can see {', '.join(labels) if labels else 'a scene'} in the image."

            return SceneDescription(
                description=description,
                frame_id=frame.frame_id,
                detections=detections,
                latency_ms=50,
            )

        import base64
        response = await self._vision_client.call(
            "aiglasses.vision.VisionService",
            "DescribeScene",
            {
                "frame_id": frame.frame_id,
                "image_data": base64.b64encode(frame.data).decode("ascii"),
                "prompt": prompt,
            },
        )

        return SceneDescription(
            description=response.get("description", ""),
            frame_id=frame.frame_id,
            detections=detections,
            latency_ms=response.get("latency_ms", 0),
        )

    async def stream_frames(
        self,
        fps: int = 10,
        format: str = "jpeg",
        quality: int = 70,
    ) -> AsyncIterator[Frame]:
        """Stream frames continuously.

        Args:
            fps: Target frames per second.
            format: Image format.
            quality: JPEG quality.

        Yields:
            Frames.

        Example:
            async for frame in app.vision.stream_frames(fps=5):
                # Process each frame
                pass
        """
        interval = 1.0 / fps

        while True:
            start_time = asyncio.get_event_loop().time()
            frame = await self.snapshot(format, quality)
            yield frame

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)


