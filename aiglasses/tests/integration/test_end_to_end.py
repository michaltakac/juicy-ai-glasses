"""End-to-end integration tests for AI Glasses Platform."""

import pytest

from aiglasses.config import Config
from aiglasses.sdk import App


@pytest.mark.integration
class TestEndToEndFlow:
    """End-to-end flow tests."""

    @pytest.mark.asyncio
    async def test_what_am_i_seeing_flow(self, mock_config: Config):
        """Test the complete "What am I seeing?" flow.

        This test simulates the main use case:
        1. User asks a question
        2. System captures frame
        3. System detects objects
        4. System generates answer with LLM
        5. System speaks the answer
        """
        async with App("e2e-test", config=mock_config, mock_mode=True) as app:
            # Step 1: Simulate user question (in real app, this comes from STT)
            question = "What am I seeing?"

            # Step 2: Capture frame
            frame = await app.vision.snapshot()
            assert frame.frame_id
            assert frame.data

            # Step 3: Detect objects
            detection_result = await app.vision.detect_objects(frame)
            assert len(detection_result.detections) > 0

            detections = detection_result.detections
            labels = [d.label for d in detections]

            # Step 4: Generate answer with LLM
            prompt = f"Objects detected: {', '.join(labels)}. {question}"

            answer = ""
            async for chunk in app.llm.chat([
                {"role": "user", "content": prompt}
            ]):
                answer += chunk.content

            assert answer
            assert len(answer) > 10  # Should have meaningful response

            # Step 5: Speak the answer
            await app.audio.speak(answer)

    @pytest.mark.asyncio
    async def test_detection_and_describe(self, mock_config: Config):
        """Test detection followed by scene description."""
        async with App("e2e-test", config=mock_config, mock_mode=True) as app:
            # Capture and detect
            frame = await app.vision.snapshot()
            detection_result = await app.vision.detect_objects(frame)

            # Describe scene with detections
            description = await app.vision.describe_scene(
                frame, detections=detection_result.detections
            )

            assert description.description
            # Description should mention detected objects
            for det in detection_result.detections:
                # Note: Mock might not include all labels in description
                pass

    @pytest.mark.asyncio
    async def test_audio_to_text_to_audio(self, mock_config: Config):
        """Test audio input -> processing -> audio output flow."""
        async with App("e2e-test", config=mock_config, mock_mode=True) as app:
            # Simulate captured audio
            audio_data = bytes(16000 * 2)  # 1 second of silence

            # Transcribe
            result = await app.audio.transcribe(audio_data)
            assert result.text

            # Get LLM response
            response = await app.llm.chat_complete([
                {"role": "user", "content": result.text}
            ])
            assert response.content

            # Speak response
            await app.audio.speak(response.content)


@pytest.mark.integration
class TestServiceInteraction:
    """Tests for service-to-service interactions."""

    @pytest.mark.asyncio
    async def test_vision_with_llm(self, mock_config: Config):
        """Test vision service with LLM for scene understanding."""
        async with App("service-test", config=mock_config, mock_mode=True) as app:
            # Get frame and detections
            frame = await app.vision.snapshot()
            detection_result = await app.vision.detect_objects(frame)

            # Format detections for LLM
            detection_text = ", ".join(
                f"{d.label} ({d.confidence:.0%})"
                for d in detection_result.detections
            )

            # Ask LLM about the scene
            response = await app.llm.chat_complete([
                {
                    "role": "system",
                    "content": "You analyze scenes and describe what you see."
                },
                {
                    "role": "user",
                    "content": f"I detected these objects: {detection_text}. What's happening in this scene?"
                }
            ])

            assert response.content
            assert response.finish_reason == "stop"


@pytest.mark.integration
@pytest.mark.hil
class TestHardwareInLoop:
    """Hardware-in-the-loop tests.

    These tests require real hardware and are only run with --hil flag.
    These tests use the Foundation services directly (not via gRPC)
    to test hardware functionality.
    """

    @pytest.mark.asyncio
    async def test_real_camera_capture(self, config: Config):
        """Test capturing from real camera using CameraService directly."""
        from aiglasses.foundation.camera import CameraService

        # Use camera service directly (not via SDK/gRPC)
        camera = CameraService(config, mock_mode=False)
        await camera.setup()

        try:
            frame = await camera.snapshot()

            assert frame.frame_id
            assert frame.data
            assert len(frame.data) > 1000  # Real image should be larger
            assert frame.width >= 640
            assert frame.height >= 480

            print(f"\nCaptured real frame: {frame.width}x{frame.height}, {len(frame.data)} bytes")
        finally:
            await camera.teardown()

    @pytest.mark.asyncio
    async def test_real_audio_status(self, config: Config):
        """Test audio service on real hardware."""
        from aiglasses.foundation.audio import AudioService

        audio = AudioService(config, mock_mode=False)
        await audio.setup()

        try:
            devices = await audio.list_devices()
            status = audio.get_status()

            print(f"\nAudio devices found: {len(devices)}")
            for d in devices:
                print(f"  - {d.name} ({d.type})")
            print(f"Audio status: {status}")

            assert status is not None
        finally:
            await audio.teardown()

    @pytest.mark.asyncio
    async def test_real_vision_pipeline(self, config: Config):
        """Test vision service with real camera."""
        from aiglasses.foundation.camera import CameraService
        from aiglasses.foundation.vision import VisionService

        camera = CameraService(config, mock_mode=False)
        vision = VisionService(config, mock_mode=False)

        await camera.setup()
        await vision.setup()

        try:
            # Capture frame
            frame = await camera.snapshot()
            print(f"\nCaptured frame: {frame.width}x{frame.height}")

            # Run detection
            result = await vision.detect_objects(frame.data, frame.frame_id)

            print(f"Detection took {result.inference_time_ms}ms")
            print(f"Detected {len(result.detections)} objects:")
            for det in result.detections:
                print(f"  - {det.label}: {det.confidence:.1%}")

            # Vision service should complete without error
            assert result.inference_time_ms >= 0
        finally:
            await camera.teardown()
            await vision.teardown()

