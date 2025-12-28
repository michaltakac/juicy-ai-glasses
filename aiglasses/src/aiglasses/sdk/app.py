"""SDK App class - main entry point for SDK applications."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from aiglasses.common.logging import get_logger, setup_logging
from aiglasses.config import Config, load_config
from aiglasses.sdk.audio import AudioAPI
from aiglasses.sdk.vision import VisionAPI
from aiglasses.sdk.llm import LLMAPI
from aiglasses.sdk.events import EventBus


class App:
    """Main application class for AI Glasses apps.

    Provides access to all SDK APIs and manages the app lifecycle.

    Example:
        async def main():
            app = App("my-app")
            await app.start()

            # Use APIs
            frame = await app.vision.snapshot()
            await app.audio.speak("Hello!")

            await app.stop()
    """

    def __init__(
        self,
        app_id: str,
        config: Config | None = None,
        mock_mode: bool | None = None,
    ) -> None:
        """Initialize the app.

        Args:
            app_id: Unique app identifier.
            config: Configuration (loaded from env/file if None).
            mock_mode: Run in mock mode (auto-detected from env if None).
        """
        self.app_id = app_id
        self.config = config or load_config()

        # Auto-detect mock mode from environment
        if mock_mode is None:
            mock_mode = os.environ.get("AIGLASSES_MOCK_MODE", "").lower() in ("1", "true", "yes")
        self.mock_mode = mock_mode or self.config.mock_mode

        # Setup logging
        setup_logging(
            level=self.config.device.log_level,
            json_output=self.config.device.mode == "production",
            service_name=app_id,
        )
        self.logger = get_logger(app_id)

        # Initialize APIs
        self._audio: AudioAPI | None = None
        self._vision: VisionAPI | None = None
        self._llm: LLMAPI | None = None
        self._events: EventBus | None = None

        # State
        self._started = False
        self._context: dict[str, Any] = {}

    @property
    def audio(self) -> AudioAPI:
        """Get the Audio API."""
        if not self._audio:
            raise RuntimeError("App not started. Call await app.start() first.")
        return self._audio

    @property
    def vision(self) -> VisionAPI:
        """Get the Vision API."""
        if not self._vision:
            raise RuntimeError("App not started. Call await app.start() first.")
        return self._vision

    @property
    def llm(self) -> LLMAPI:
        """Get the LLM API."""
        if not self._llm:
            raise RuntimeError("App not started. Call await app.start() first.")
        return self._llm

    @property
    def events(self) -> EventBus:
        """Get the Event Bus."""
        if not self._events:
            raise RuntimeError("App not started. Call await app.start() first.")
        return self._events

    async def start(self) -> None:
        """Start the app and connect to services.

        This must be called before using any APIs.
        """
        if self._started:
            return

        self.logger.info("app_starting", app_id=self.app_id, mock_mode=self.mock_mode)

        # Initialize APIs
        self._audio = AudioAPI(self.config, self.mock_mode)
        self._vision = VisionAPI(self.config, self.mock_mode)
        self._llm = LLMAPI(self.config, self.mock_mode)
        self._events = EventBus()

        # Connect to services
        await self._audio.connect()
        await self._vision.connect()
        await self._llm.connect()

        self._started = True
        self.logger.info("app_started", app_id=self.app_id)

    async def stop(self) -> None:
        """Stop the app and disconnect from services."""
        if not self._started:
            return

        self.logger.info("app_stopping", app_id=self.app_id)

        # Disconnect from services
        if self._audio:
            await self._audio.disconnect()
        if self._vision:
            await self._vision.disconnect()
        if self._llm:
            await self._llm.disconnect()

        self._started = False
        self.logger.info("app_stopped", app_id=self.app_id)

    async def __aenter__(self) -> App:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value for the app."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)


def create_app(app_id: str, **kwargs: Any) -> App:
    """Create a new app instance.

    This is a convenience function for creating apps.

    Args:
        app_id: Unique app identifier.
        **kwargs: Additional arguments passed to App().

    Returns:
        App instance (not started).
    """
    return App(app_id, **kwargs)


