"""SDK Audio API - voice input and output."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Awaitable

from aiglasses.common.logging import get_logger
from aiglasses.common.grpc_utils import GrpcClient
from aiglasses.config import Config


@dataclass
class AudioChunk:
    """Audio data chunk."""

    data: bytes
    timestamp_ms: int
    sample_rate: int
    channels: int
    format: str
    sequence: int


@dataclass
class TranscribeResult:
    """Speech-to-text result."""

    text: str
    is_final: bool
    confidence: float
    language: str


@dataclass
class WakeEvent:
    """Wake word detection event."""

    wake_word: str
    confidence: float
    timestamp: float


class AudioAPI:
    """Audio API for voice input and output.

    Provides methods for:
    - Microphone capture
    - Speech-to-text transcription
    - Text-to-speech synthesis
    - Wake word detection
    """

    def __init__(self, config: Config, mock_mode: bool = False) -> None:
        """Initialize Audio API.

        Args:
            config: Configuration.
            mock_mode: Run in mock mode.
        """
        self.config = config
        self.mock_mode = mock_mode
        self.logger = get_logger("sdk.audio")

        self._audio_client: GrpcClient | None = None
        self._speech_client: GrpcClient | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to audio and speech services."""
        if self._connected:
            return

        if not self.mock_mode:
            self._audio_client = GrpcClient("localhost", self.config.ports.audio)
            self._speech_client = GrpcClient("localhost", self.config.ports.speech)
            await self._audio_client.connect()
            await self._speech_client.connect()

        self._connected = True
        self.logger.debug("audio_api_connected")

    async def disconnect(self) -> None:
        """Disconnect from services."""
        if self._audio_client:
            await self._audio_client.close()
        if self._speech_client:
            await self._speech_client.close()
        self._connected = False

    async def listen(
        self,
        sample_rate: int | None = None,
        channels: int | None = None,
        chunk_size_ms: int | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Listen to microphone input.

        Args:
            sample_rate: Sample rate in Hz (default from config).
            channels: Number of channels (default 1).
            chunk_size_ms: Chunk size in milliseconds.

        Yields:
            Audio chunks from microphone.

        Example:
            async for chunk in app.audio.listen():
                # Process audio chunk
                print(f"Got {len(chunk.data)} bytes")
        """
        sample_rate = sample_rate or self.config.audio.sample_rate
        channels = channels or self.config.audio.channels
        chunk_size_ms = chunk_size_ms or self.config.audio.chunk_size_ms

        if self.mock_mode:
            # Generate mock audio chunks
            sequence = 0
            chunk_samples = int(sample_rate * chunk_size_ms / 1000)
            while True:
                await asyncio.sleep(chunk_size_ms / 1000)
                yield AudioChunk(
                    data=bytes(chunk_samples * channels * 2),
                    timestamp_ms=int(asyncio.get_event_loop().time() * 1000),
                    sample_rate=sample_rate,
                    channels=channels,
                    format="pcm_s16le",
                    sequence=sequence,
                )
                sequence += 1
        else:
            async for chunk_data in self._audio_client.call_stream(
                "aiglasses.audio.AudioService",
                "OpenMic",
                {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "chunk_size_ms": chunk_size_ms,
                },
            ):
                import base64
                yield AudioChunk(
                    data=base64.b64decode(chunk_data.get("data", "")),
                    timestamp_ms=chunk_data.get("timestamp_ms", 0),
                    sample_rate=chunk_data.get("sample_rate", sample_rate),
                    channels=chunk_data.get("channels", channels),
                    format=chunk_data.get("format", "pcm_s16le"),
                    sequence=chunk_data.get("sequence", 0),
                )

    async def transcribe(
        self,
        audio_data: bytes,
        format: str = "wav",
        language: str | None = None,
    ) -> TranscribeResult:
        """Transcribe audio to text.

        Args:
            audio_data: Audio data bytes.
            format: Audio format ("wav", "mp3", "opus").
            language: Language code or "auto".

        Returns:
            Transcription result.

        Example:
            result = await app.audio.transcribe(audio_bytes)
            print(f"You said: {result.text}")
        """
        language = language or self.config.speech.stt_language

        if self.mock_mode:
            await asyncio.sleep(0.1)
            return TranscribeResult(
                text="What am I seeing?",
                is_final=True,
                confidence=0.95,
                language=language,
            )

        import base64
        response = await self._speech_client.call(
            "aiglasses.speech.SpeechService",
            "TranscribeBytes",
            {
                "audio_data": base64.b64encode(audio_data).decode("ascii"),
                "format": format,
                "language": language,
            },
        )

        return TranscribeResult(
            text=response.get("text", ""),
            is_final=response.get("is_final", True),
            confidence=response.get("confidence", 0.0),
            language=response.get("language", language),
        )

    async def speak(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        wait: bool = True,
    ) -> None:
        """Speak text using text-to-speech.

        Args:
            text: Text to speak.
            voice: Voice ID (default from config).
            speed: Speech speed (0.5-2.0).
            wait: Wait for speech to complete.

        Example:
            await app.audio.speak("Hello, I can see a person in front of you.")
        """
        voice = voice or self.config.speech.tts_voice
        speed = speed or self.config.speech.tts_speed

        if self.mock_mode:
            self.logger.info("mock_speak", text=text)
            if wait:
                # Simulate speech duration (~50ms per character)
                await asyncio.sleep(len(text) * 0.05)
            return

        # Synthesize speech
        import base64
        response = await self._speech_client.call(
            "aiglasses.speech.SpeechService",
            "SynthesizeComplete",
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "format": "wav",
            },
        )

        audio_data = base64.b64decode(response.get("audio_data", ""))

        # Play audio
        await self._audio_client.call(
            "aiglasses.audio.AudioService",
            "PlayBytes",
            {
                "audio_data": base64.b64encode(audio_data).decode("ascii"),
                "format": "wav",
                "wait": wait,
            },
        )

    async def on_wake_word(
        self,
        wake_words: list[str] | None = None,
        sensitivity: float | None = None,
    ) -> AsyncIterator[WakeEvent]:
        """Listen for wake word.

        Args:
            wake_words: Wake words to detect (default from config).
            sensitivity: Detection sensitivity (0.0-1.0).

        Yields:
            Wake word events when detected.

        Example:
            async for wake in app.audio.on_wake_word():
                print(f"Wake word detected: {wake.wake_word}")
                # Handle user interaction
        """
        wake_words = wake_words or self.config.speech.wake_words
        sensitivity = sensitivity or self.config.speech.wake_sensitivity

        if self.mock_mode:
            # Generate mock wake events periodically
            while True:
                await asyncio.sleep(5.0)  # Wake every 5 seconds in mock mode
                yield WakeEvent(
                    wake_word=wake_words[0] if wake_words else "hey glasses",
                    confidence=0.9,
                    timestamp=asyncio.get_event_loop().time(),
                )
        else:
            # Real implementation would use speech service
            # For now, implement a simple polling approach
            async for chunk in self.listen():
                # In a real implementation, this would send audio to wake detection
                # For now, just yield events periodically
                pass


