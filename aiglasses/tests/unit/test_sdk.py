"""Tests for SDK module."""

import pytest

from aiglasses.config import Config
from aiglasses.sdk import App
from aiglasses.sdk.manifest import AppManifest, Permission, create_manifest


class TestApp:
    """Tests for App class."""

    @pytest.mark.asyncio
    async def test_app_lifecycle(self, mock_config: Config):
        """Test app start and stop."""
        app = App("test-app", config=mock_config, mock_mode=True)

        # Should not be started
        with pytest.raises(RuntimeError):
            _ = app.audio

        # Start app
        await app.start()

        # APIs should be available
        assert app.audio is not None
        assert app.vision is not None
        assert app.llm is not None
        assert app.events is not None

        # Stop app
        await app.stop()

    @pytest.mark.asyncio
    async def test_app_context_manager(self, mock_config: Config):
        """Test app as context manager."""
        async with App("test-app", config=mock_config, mock_mode=True) as app:
            assert app.audio is not None
            assert app.vision is not None

    @pytest.mark.asyncio
    async def test_app_context_storage(self, mock_config: Config):
        """Test app context storage."""
        app = App("test-app", config=mock_config, mock_mode=True)
        await app.start()

        app.set_context("key", "value")
        assert app.get_context("key") == "value"
        assert app.get_context("missing") is None
        assert app.get_context("missing", "default") == "default"

        await app.stop()


class TestAppManifest:
    """Tests for AppManifest class."""

    def test_manifest_creation(self):
        """Test creating manifest."""
        manifest = AppManifest(
            name="Test App",
            version="1.0.0",
            description="Test description",
            entrypoint="python main.py",
            required_permissions=[Permission.CAMERA, Permission.MICROPHONE],
        )

        assert manifest.name == "Test App"
        assert manifest.version == "1.0.0"
        assert Permission.CAMERA in manifest.required_permissions
        assert Permission.MICROPHONE in manifest.required_permissions

    def test_manifest_validate_permissions(self):
        """Test permission validation."""
        manifest = AppManifest(
            name="Test",
            version="1.0.0",
            required_permissions=[Permission.CAMERA, Permission.MICROPHONE],
        )

        # All required permissions granted
        missing = manifest.validate_permissions([Permission.CAMERA, Permission.MICROPHONE])
        assert len(missing) == 0

        # Missing some permissions
        missing = manifest.validate_permissions([Permission.CAMERA])
        assert Permission.MICROPHONE in missing

    def test_manifest_has_permission(self):
        """Test has_permission check."""
        manifest = AppManifest(
            name="Test",
            version="1.0.0",
            required_permissions=[Permission.CAMERA],
            optional_permissions=[Permission.STORAGE],
        )

        assert manifest.has_permission(Permission.CAMERA)
        assert manifest.has_permission(Permission.STORAGE)
        assert not manifest.has_permission(Permission.BLUETOOTH)

    def test_create_manifest_helper(self):
        """Test create_manifest helper function."""
        manifest = create_manifest(
            name="My App",
            version="2.0.0",
            permissions=["camera", "network"],
        )

        assert manifest.name == "My App"
        assert manifest.version == "2.0.0"
        assert Permission.CAMERA in manifest.required_permissions
        assert Permission.NETWORK in manifest.required_permissions


class TestAudioAPI:
    """Tests for Audio API."""

    @pytest.mark.asyncio
    async def test_speak(self, sdk_app: App):
        """Test speak functionality."""
        # Should not raise in mock mode
        await sdk_app.audio.speak("Hello world")

    @pytest.mark.asyncio
    async def test_transcribe(self, sdk_app: App):
        """Test transcribe functionality."""
        # Mock audio data
        audio_data = bytes(16000 * 2)  # 1 second of silence

        result = await sdk_app.audio.transcribe(audio_data)

        assert result.text  # Should have some text in mock mode
        assert result.is_final
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_listen(self, sdk_app: App):
        """Test listen functionality."""
        chunk_count = 0
        async for chunk in sdk_app.audio.listen():
            chunk_count += 1
            assert chunk.data is not None
            assert chunk.sample_rate > 0
            if chunk_count >= 3:
                break

        assert chunk_count == 3


class TestVisionAPI:
    """Tests for Vision API."""

    @pytest.mark.asyncio
    async def test_snapshot(self, sdk_app: App):
        """Test snapshot functionality."""
        frame = await sdk_app.vision.snapshot()

        assert frame.frame_id
        assert frame.data
        assert frame.width > 0
        assert frame.height > 0

    @pytest.mark.asyncio
    async def test_detect_objects(self, sdk_app: App):
        """Test object detection."""
        frame = await sdk_app.vision.snapshot()
        result = await sdk_app.vision.detect_objects(frame)

        assert result.frame_id == frame.frame_id
        assert len(result.detections) > 0  # Mock should return detections

        for detection in result.detections:
            assert detection.label
            assert 0 <= detection.confidence <= 1
            assert detection.bbox is not None

    @pytest.mark.asyncio
    async def test_describe_scene(self, sdk_app: App):
        """Test scene description."""
        frame = await sdk_app.vision.snapshot()
        result = await sdk_app.vision.describe_scene(frame)

        assert result.description
        assert result.frame_id == frame.frame_id


class TestLLMAPI:
    """Tests for LLM API."""

    @pytest.mark.asyncio
    async def test_chat_streaming(self, sdk_app: App):
        """Test streaming chat."""
        content = ""
        async for chunk in sdk_app.llm.chat([
            {"role": "user", "content": "Hello"}
        ]):
            content += chunk.content

        assert content  # Should have some response

    @pytest.mark.asyncio
    async def test_chat_complete(self, sdk_app: App):
        """Test non-streaming chat."""
        response = await sdk_app.llm.chat_complete([
            {"role": "user", "content": "What do you see?"}
        ])

        assert response.content
        assert response.finish_reason

    @pytest.mark.asyncio
    async def test_ask_helper(self, sdk_app: App):
        """Test ask helper method."""
        answer = await sdk_app.llm.ask(
            "What do you see?",
            detections=[{"label": "person"}, {"label": "laptop"}],
        )

        assert answer
        assert isinstance(answer, str)


