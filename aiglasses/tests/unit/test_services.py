"""Tests for Foundation services."""

import pytest

from aiglasses.config import Config


class TestCameraService:
    """Tests for Camera service."""

    @pytest.mark.asyncio
    async def test_snapshot(self, camera_service):
        """Test capturing a snapshot."""
        frame = await camera_service.snapshot()

        assert frame.frame_id
        assert frame.data
        assert frame.width > 0
        assert frame.height > 0

    @pytest.mark.asyncio
    async def test_get_frame(self, camera_service):
        """Test retrieving cached frame."""
        frame = await camera_service.snapshot()
        retrieved = await camera_service.get_frame(frame.frame_id)

        assert retrieved is not None
        assert retrieved.frame_id == frame.frame_id

    @pytest.mark.asyncio
    async def test_get_status(self, camera_service):
        """Test getting camera status."""
        status = camera_service.get_status()

        assert "available" in status
        assert "state" in status


class TestAudioService:
    """Tests for Audio service."""

    @pytest.mark.asyncio
    async def test_list_devices(self, audio_service):
        """Test listing audio devices."""
        devices = await audio_service.list_devices()

        assert isinstance(devices, list)
        # Mock mode should have at least one device
        assert len(devices) > 0

    @pytest.mark.asyncio
    async def test_get_status(self, audio_service):
        """Test getting audio status."""
        status = audio_service.get_status()

        assert "available" in status


class TestSpeechService:
    """Tests for Speech service."""

    @pytest.mark.asyncio
    async def test_transcribe_bytes(self, speech_service):
        """Test transcribing audio bytes."""
        # Mock audio data
        audio_data = bytes(16000 * 2)

        result = await speech_service.transcribe_bytes(audio_data, format="raw")

        assert result.text
        assert result.is_final

    @pytest.mark.asyncio
    async def test_synthesize_complete(self, speech_service):
        """Test synthesizing speech."""
        result = await speech_service.synthesize_complete("Hello world")

        assert result.audio_data
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_get_status(self, speech_service):
        """Test getting speech status."""
        status = speech_service.get_status()

        assert "stt_available" in status
        assert "tts_available" in status


class TestVisionService:
    """Tests for Vision service."""

    @pytest.mark.asyncio
    async def test_detect_objects(self, vision_service, mock_image_bytes: bytes):
        """Test object detection."""
        result = await vision_service.detect_objects(mock_image_bytes)

        assert result.detections is not None
        assert result.inference_time_ms >= 0

    @pytest.mark.asyncio
    async def test_describe_scene(self, vision_service, mock_image_bytes: bytes):
        """Test scene description."""
        result = await vision_service.describe_scene(mock_image_bytes)

        assert result.description
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_get_status(self, vision_service):
        """Test getting vision status."""
        status = vision_service.get_status()

        assert "available" in status


class TestLLMGateway:
    """Tests for LLM Gateway service."""

    @pytest.mark.asyncio
    async def test_chat(self, llm_gateway):
        """Test chat completion."""
        from aiglasses.foundation.llm_gateway.service import Message

        messages = [
            Message(role="user", content="Hello"),
        ]

        response_text = ""
        async for chunk in llm_gateway.chat(messages):
            response_text += chunk.content

        assert response_text

    @pytest.mark.asyncio
    async def test_get_status(self, llm_gateway):
        """Test getting gateway status."""
        status = llm_gateway.get_status()

        assert "available" in status
        assert "active_provider" in status


class TestStorageService:
    """Tests for Storage service."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, storage_service):
        """Test storing and retrieving data."""
        data = b"test data"
        key = "test-key"

        item = await storage_service.store(key, data)
        assert item.key == key

        retrieved = await storage_service.retrieve(key)
        assert retrieved is not None
        assert retrieved.data == data

    @pytest.mark.asyncio
    async def test_delete(self, storage_service):
        """Test deleting data."""
        data = b"test data"
        key = "test-key-delete"

        await storage_service.store(key, data)
        deleted = await storage_service.delete(key)
        assert deleted

        retrieved = await storage_service.retrieve(key)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_purge(self, storage_service):
        """Test purging all data."""
        # Store some items
        await storage_service.store("key1", b"data1")
        await storage_service.store("key2", b"data2")

        items_deleted, bytes_freed = await storage_service.purge()

        assert items_deleted >= 2
        assert bytes_freed > 0

    @pytest.mark.asyncio
    async def test_get_status(self, storage_service):
        """Test getting storage status."""
        status = storage_service.get_status()

        assert "available" in status
        assert "total_items" in status


