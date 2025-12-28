"""Pytest configuration and fixtures for AI Glasses tests."""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator
from unittest.mock import MagicMock, AsyncMock

import pytest

from aiglasses.config import Config, load_config


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--mock",
        action="store_true",
        default=True,
        help="Run tests with mock services (default)",
    )
    parser.addoption(
        "--hil",
        action="store_true",
        default=False,
        help="Run hardware-in-the-loop tests (requires real hardware)",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure test markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "hil: Hardware-in-the-loop tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection based on options."""
    # Skip HIL tests unless --hil flag is set
    if not config.getoption("--hil"):
        skip_hil = pytest.mark.skip(reason="Need --hil option to run")
        for item in items:
            if "hil" in item.keywords:
                item.add_marker(skip_hil)

    # Skip slow tests unless --slow flag is set
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Need --slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_mode(request: pytest.FixtureRequest) -> bool:
    """Get mock mode from command line or default.
    
    For unit tests, always use mock mode.
    For integration tests with --hil, use real hardware.
    """
    # Check if this is a HIL-marked test
    if "hil" in request.keywords:
        return not request.config.getoption("--hil")
    # Unit tests always use mock mode
    return True


@pytest.fixture
def config(mock_mode: bool) -> Config:
    """Get test configuration."""
    cfg = Config()
    cfg.mock_mode = mock_mode
    cfg.device.mode = "development"
    cfg.device.log_level = "DEBUG"
    return cfg


@pytest.fixture
def hil_config(request: pytest.FixtureRequest) -> Config:
    """Get configuration for HIL tests (real hardware)."""
    cfg = Config()
    cfg.mock_mode = not request.config.getoption("--hil")
    cfg.device.mode = "development"
    cfg.device.log_level = "DEBUG"
    return cfg


@pytest.fixture
def mock_config() -> Config:
    """Get mock configuration."""
    cfg = Config()
    cfg.mock_mode = True
    cfg.device.mode = "development"
    return cfg


# Service fixtures


@pytest.fixture
async def device_manager(config: Config):
    """Create Device Manager service for testing."""
    from aiglasses.foundation.device_manager import DeviceManagerService

    service = DeviceManagerService(config, mock_mode=config.mock_mode)
    await service.setup()
    yield service
    await service.teardown()


@pytest.fixture
async def camera_service(config: Config):
    """Create Camera service for testing."""
    from aiglasses.foundation.camera import CameraService

    service = CameraService(config, mock_mode=config.mock_mode)
    await service.setup()
    yield service
    await service.teardown()


@pytest.fixture
async def audio_service(config: Config):
    """Create Audio service for testing."""
    from aiglasses.foundation.audio import AudioService

    service = AudioService(config, mock_mode=config.mock_mode)
    await service.setup()
    yield service
    await service.teardown()


@pytest.fixture
async def speech_service(config: Config):
    """Create Speech service for testing."""
    from aiglasses.foundation.speech import SpeechService

    service = SpeechService(config, mock_mode=config.mock_mode)
    await service.setup()
    yield service
    await service.teardown()


@pytest.fixture
async def vision_service(config: Config):
    """Create Vision service for testing."""
    from aiglasses.foundation.vision import VisionService

    service = VisionService(config, mock_mode=config.mock_mode)
    await service.setup()
    yield service
    await service.teardown()


@pytest.fixture
async def llm_gateway(config: Config):
    """Create LLM Gateway service for testing."""
    from aiglasses.foundation.llm_gateway import LLMGatewayService

    service = LLMGatewayService(config, mock_mode=config.mock_mode)
    await service.setup()
    yield service
    await service.teardown()


@pytest.fixture
async def storage_service(config: Config):
    """Create Storage service for testing."""
    from aiglasses.foundation.storage import StorageService

    service = StorageService(config, mock_mode=config.mock_mode)
    await service.setup()
    yield service
    await service.teardown()


# SDK fixtures


@pytest.fixture
async def sdk_app(mock_config: Config):
    """Create SDK App for testing."""
    from aiglasses.sdk import App

    app = App("test-app", config=mock_config, mock_mode=True)
    await app.start()
    yield app
    await app.stop()


# Mock fixtures


@pytest.fixture
def mock_audio_chunk() -> bytes:
    """Create mock audio chunk."""
    # 100ms of 16kHz mono 16-bit PCM silence
    return bytes(16000 * 2 // 10)


@pytest.fixture
def mock_image_bytes() -> bytes:
    """Create mock JPEG image."""
    from PIL import Image
    import io

    img = Image.new("RGB", (640, 480), color=(73, 109, 137))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()

