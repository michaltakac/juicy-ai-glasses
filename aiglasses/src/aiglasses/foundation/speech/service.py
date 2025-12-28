"""Speech service implementation."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Awaitable

from grpc import aio as grpc_aio

from aiglasses.common import BaseService, get_logger
from aiglasses.common.events import Event, get_event_bus
from aiglasses.config import Config


@dataclass
class TranscribeResult:
    """Speech-to-text result."""

    text: str
    is_final: bool
    confidence: float
    language: str
    latency_ms: int
    words: list[dict] = field(default_factory=list)


@dataclass
class SynthesizeResult:
    """Text-to-speech result."""

    audio_data: bytes
    format: str
    sample_rate: int
    duration_ms: int
    latency_ms: int


@dataclass
class WakeEvent:
    """Wake word detection event."""

    wake_word: str
    confidence: float
    timestamp: float
    audio_after_wake: bytes = b""


class SpeechBackend:
    """Abstract speech backend."""

    async def setup(self) -> None:
        """Setup speech backend."""
        pass

    async def teardown(self) -> None:
        """Teardown speech backend."""
        pass

    async def transcribe(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000,
        language: str = "en",
    ) -> AsyncIterator[TranscribeResult]:
        """Transcribe audio to text."""
        raise NotImplementedError

    async def transcribe_bytes(
        self,
        audio_data: bytes,
        format: str = "wav",
        language: str = "en",
    ) -> TranscribeResult:
        """Transcribe audio bytes to text."""
        raise NotImplementedError

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        format: str = "wav",
    ) -> AsyncIterator[bytes]:
        """Synthesize text to speech (streaming)."""
        raise NotImplementedError

    async def synthesize_complete(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        format: str = "wav",
    ) -> SynthesizeResult:
        """Synthesize text to speech (complete)."""
        raise NotImplementedError

    async def detect_wake(
        self,
        audio_chunks: AsyncIterator[bytes],
        wake_words: list[str],
        sensitivity: float = 0.5,
    ) -> AsyncIterator[WakeEvent]:
        """Detect wake words in audio stream."""
        raise NotImplementedError

    def get_status(self) -> dict:
        """Get speech backend status."""
        raise NotImplementedError


class MockSpeechBackend(SpeechBackend):
    """Mock speech backend for testing."""

    def __init__(self) -> None:
        self._stt_available = True
        self._tts_available = True
        self._wake_detection_active = False

    async def setup(self) -> None:
        """Setup mock speech."""
        pass

    async def teardown(self) -> None:
        """Teardown mock speech."""
        pass

    async def transcribe(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000,
        language: str = "en",
    ) -> AsyncIterator[TranscribeResult]:
        """Mock transcription."""
        # Consume some audio chunks
        chunk_count = 0
        async for _ in audio_chunks:
            chunk_count += 1
            if chunk_count >= 5:  # After 5 chunks, emit a result
                yield TranscribeResult(
                    text="What am I seeing?",
                    is_final=True,
                    confidence=0.95,
                    language=language,
                    latency_ms=100,
                )
                break
            elif chunk_count == 3:
                yield TranscribeResult(
                    text="What am I",
                    is_final=False,
                    confidence=0.8,
                    language=language,
                    latency_ms=50,
                )

    async def transcribe_bytes(
        self,
        audio_data: bytes,
        format: str = "wav",
        language: str = "en",
    ) -> TranscribeResult:
        """Mock transcription from bytes."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return TranscribeResult(
            text="What am I seeing?",
            is_final=True,
            confidence=0.95,
            language=language,
            latency_ms=100,
        )

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        format: str = "wav",
    ) -> AsyncIterator[bytes]:
        """Mock synthesis (streaming)."""
        # Generate mock audio in chunks
        result = await self.synthesize_complete(text, voice, speed, format)
        chunk_size = 4096
        for i in range(0, len(result.audio_data), chunk_size):
            yield result.audio_data[i : i + chunk_size]
            await asyncio.sleep(0.01)

    async def synthesize_complete(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        format: str = "wav",
    ) -> SynthesizeResult:
        """Mock synthesis (complete)."""
        start_time = time.time()

        # Generate silent WAV audio
        sample_rate = 22050
        duration_seconds = len(text) * 0.05  # ~50ms per character
        num_samples = int(sample_rate * duration_seconds)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(num_samples * 2))

        audio_data = buffer.getvalue()
        latency_ms = int((time.time() - start_time) * 1000)

        return SynthesizeResult(
            audio_data=audio_data,
            format=format,
            sample_rate=sample_rate,
            duration_ms=int(duration_seconds * 1000),
            latency_ms=latency_ms,
        )

    async def detect_wake(
        self,
        audio_chunks: AsyncIterator[bytes],
        wake_words: list[str],
        sensitivity: float = 0.5,
    ) -> AsyncIterator[WakeEvent]:
        """Mock wake word detection."""
        self._wake_detection_active = True
        try:
            chunk_count = 0
            async for chunk in audio_chunks:
                chunk_count += 1
                # Simulate wake word detected every 50 chunks
                if chunk_count % 50 == 0:
                    yield WakeEvent(
                        wake_word=wake_words[0] if wake_words else "hey glasses",
                        confidence=0.9,
                        timestamp=time.time(),
                        audio_after_wake=chunk,
                    )
        finally:
            self._wake_detection_active = False

    def get_status(self) -> dict:
        """Get mock speech status."""
        return {
            "stt_available": self._stt_available,
            "tts_available": self._tts_available,
            "wake_detection_active": self._wake_detection_active,
            "current_stt_model": "mock",
            "current_tts_voice": "mock",
        }


class WhisperSpeechBackend(SpeechBackend):
    """Speech backend using Whisper for STT and Piper for TTS."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._whisper_model = None
        self._piper_voice = None
        self._wake_detection_active = False
        self.logger = get_logger("whisper_speech_backend")

    async def setup(self) -> None:
        """Setup Whisper/Piper speech."""
        try:
            from faster_whisper import WhisperModel

            model_size = self.config.speech.stt_model
            self._whisper_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
            )
            self.logger.info("whisper_model_loaded", model=model_size)
        except ImportError:
            self.logger.warning("faster_whisper_not_available")
        except Exception as e:
            self.logger.exception("whisper_setup_failed", error=str(e))

    async def teardown(self) -> None:
        """Teardown speech backend."""
        self._whisper_model = None

    async def transcribe(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000,
        language: str = "en",
    ) -> AsyncIterator[TranscribeResult]:
        """Transcribe audio using Whisper."""
        if not self._whisper_model:
            return

        # Collect audio chunks
        audio_buffer = bytearray()
        async for chunk in audio_chunks:
            audio_buffer.extend(chunk)

            # Process every ~1 second of audio
            if len(audio_buffer) >= sample_rate * 2:  # 16-bit = 2 bytes/sample
                start_time = time.time()

                # Convert to numpy array
                import numpy as np

                audio_array = np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32)
                audio_array /= 32768.0  # Normalize to [-1, 1]

                # Run transcription
                segments, info = self._whisper_model.transcribe(
                    audio_array,
                    language=language if language != "auto" else None,
                    vad_filter=True,
                )

                text = " ".join(segment.text for segment in segments)
                latency_ms = int((time.time() - start_time) * 1000)

                if text.strip():
                    yield TranscribeResult(
                        text=text.strip(),
                        is_final=True,
                        confidence=0.9,
                        language=info.language,
                        latency_ms=latency_ms,
                    )

                audio_buffer.clear()

    async def transcribe_bytes(
        self,
        audio_data: bytes,
        format: str = "wav",
        language: str = "en",
    ) -> TranscribeResult:
        """Transcribe audio bytes using Whisper."""
        if not self._whisper_model:
            raise RuntimeError("Whisper model not loaded")

        start_time = time.time()

        # Load audio
        import numpy as np

        if format == "wav":
            with io.BytesIO(audio_data) as f:
                with wave.open(f, "rb") as wf:
                    audio_array = np.frombuffer(
                        wf.readframes(wf.getnframes()),
                        dtype=np.int16,
                    ).astype(np.float32)
                    audio_array /= 32768.0
        else:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array /= 32768.0

        # Run transcription
        segments, info = self._whisper_model.transcribe(
            audio_array,
            language=language if language != "auto" else None,
            vad_filter=True,
        )

        text = " ".join(segment.text for segment in segments)
        latency_ms = int((time.time() - start_time) * 1000)

        return TranscribeResult(
            text=text.strip(),
            is_final=True,
            confidence=0.9,
            language=info.language,
            latency_ms=latency_ms,
        )

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        format: str = "wav",
    ) -> AsyncIterator[bytes]:
        """Synthesize using Piper (streaming)."""
        result = await self.synthesize_complete(text, voice, speed, format)
        chunk_size = 4096
        for i in range(0, len(result.audio_data), chunk_size):
            yield result.audio_data[i : i + chunk_size]

    async def synthesize_complete(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        format: str = "wav",
    ) -> SynthesizeResult:
        """Synthesize using Piper or espeak fallback."""
        start_time = time.time()

        try:
            # Try Piper TTS first
            proc = await asyncio.create_subprocess_exec(
                "piper",
                "--model", voice if voice != "default" else "en_US-lessac-medium",
                "--output_file", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(text.encode())

            if proc.returncode == 0:
                audio_data = stdout
            else:
                raise RuntimeError(f"Piper failed: {stderr.decode()}")

        except (FileNotFoundError, RuntimeError):
            # Fallback to espeak
            self.logger.warning("piper_not_available_using_espeak")

            proc = await asyncio.create_subprocess_exec(
                "espeak-ng",
                "-v", "en",
                "-s", str(int(175 * speed)),
                "--stdout",
                text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            audio_data = stdout

        latency_ms = int((time.time() - start_time) * 1000)

        # Parse WAV to get duration
        try:
            with io.BytesIO(audio_data) as f:
                with wave.open(f, "rb") as wf:
                    duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)
                    sample_rate = wf.getframerate()
        except Exception:
            duration_ms = 0
            sample_rate = 22050

        return SynthesizeResult(
            audio_data=audio_data,
            format=format,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            latency_ms=latency_ms,
        )

    async def detect_wake(
        self,
        audio_chunks: AsyncIterator[bytes],
        wake_words: list[str],
        sensitivity: float = 0.5,
    ) -> AsyncIterator[WakeEvent]:
        """Detect wake words using simple keyword spotting.

        Note: For production, use Porcupine or similar optimized wake word detection.
        """
        self._wake_detection_active = True
        try:
            # Simple implementation: buffer audio and run periodic transcription
            audio_buffer = bytearray()
            sample_rate = 16000

            async for chunk in audio_chunks:
                audio_buffer.extend(chunk)

                # Check every ~2 seconds of audio
                if len(audio_buffer) >= sample_rate * 2 * 2:
                    try:
                        result = await self.transcribe_bytes(bytes(audio_buffer), "raw", "en")
                        text_lower = result.text.lower()

                        for wake_word in wake_words:
                            if wake_word.lower() in text_lower:
                                yield WakeEvent(
                                    wake_word=wake_word,
                                    confidence=result.confidence,
                                    timestamp=time.time(),
                                )
                                break
                    except Exception:
                        pass

                    audio_buffer.clear()
        finally:
            self._wake_detection_active = False

    def get_status(self) -> dict:
        """Get speech backend status."""
        return {
            "stt_available": self._whisper_model is not None,
            "tts_available": True,  # espeak fallback always available
            "wake_detection_active": self._wake_detection_active,
            "current_stt_model": self.config.speech.stt_model,
            "current_tts_voice": self.config.speech.tts_voice,
        }


class SpeechService(BaseService):
    """Speech service.

    Responsibilities:
    - Speech-to-text (STT)
    - Text-to-speech (TTS)
    - Wake word detection
    """

    def __init__(self, config: Config | None = None, mock_mode: bool = False) -> None:
        super().__init__("speech", config, mock_mode)
        self._backend: SpeechBackend | None = None
        self._event_bus = get_event_bus()
        self._wake_callbacks: list[Callable[[WakeEvent], Awaitable[None]]] = []
        self._wake_task: asyncio.Task | None = None

    @property
    def port(self) -> int:
        return self.config.ports.speech

    async def setup(self) -> None:
        """Setup Speech service."""
        self.logger.info("speech_service_setup", mock_mode=self.mock_mode)

        if self.mock_mode:
            self._backend = MockSpeechBackend()
        else:
            self._backend = WhisperSpeechBackend(self.config)

        await self._backend.setup()

    async def teardown(self) -> None:
        """Teardown Speech service."""
        self.logger.info("speech_service_teardown")

        if self._wake_task:
            self._wake_task.cancel()
            try:
                await self._wake_task
            except asyncio.CancelledError:
                pass

        if self._backend:
            await self._backend.teardown()

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services."""
        from aiglasses.foundation.speech.grpc_servicer import SpeechServicer, add_servicer

        servicer = SpeechServicer(self)
        add_servicer(server, servicer)

    async def transcribe(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> AsyncIterator[TranscribeResult]:
        """Transcribe audio to text.

        Args:
            audio_chunks: Audio data chunks.
            sample_rate: Audio sample rate.
            language: Language code or "auto".

        Yields:
            Transcription results.
        """
        if not self._backend:
            return

        language = language or self.config.speech.stt_language

        async for result in self._backend.transcribe(audio_chunks, sample_rate, language):
            yield result

    async def transcribe_bytes(
        self,
        audio_data: bytes,
        format: str = "wav",
        language: str | None = None,
    ) -> TranscribeResult:
        """Transcribe audio bytes to text.

        Args:
            audio_data: Audio data.
            format: Audio format.
            language: Language code.

        Returns:
            Transcription result.
        """
        if not self._backend:
            raise RuntimeError("Speech backend not initialized")

        language = language or self.config.speech.stt_language
        return await self._backend.transcribe_bytes(audio_data, format, language)

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        format: str = "wav",
    ) -> AsyncIterator[bytes]:
        """Synthesize text to speech (streaming).

        Args:
            text: Text to synthesize.
            voice: Voice ID.
            speed: Speech speed.
            format: Output format.

        Yields:
            Audio data chunks.
        """
        if not self._backend:
            return

        voice = voice or self.config.speech.tts_voice
        speed = speed or self.config.speech.tts_speed

        async for chunk in self._backend.synthesize(text, voice, speed, format):
            yield chunk

    async def synthesize_complete(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        format: str = "wav",
    ) -> SynthesizeResult:
        """Synthesize text to speech (complete).

        Args:
            text: Text to synthesize.
            voice: Voice ID.
            speed: Speech speed.
            format: Output format.

        Returns:
            Synthesis result with audio data.
        """
        if not self._backend:
            raise RuntimeError("Speech backend not initialized")

        voice = voice or self.config.speech.tts_voice
        speed = speed or self.config.speech.tts_speed

        return await self._backend.synthesize_complete(text, voice, speed, format)

    async def start_wake_detection(
        self,
        audio_source: AsyncIterator[bytes],
        wake_words: list[str] | None = None,
        sensitivity: float | None = None,
    ) -> AsyncIterator[WakeEvent]:
        """Start wake word detection.

        Args:
            audio_source: Audio data source.
            wake_words: Wake words to detect.
            sensitivity: Detection sensitivity.

        Yields:
            Wake word events.
        """
        if not self._backend:
            return

        wake_words = wake_words or self.config.speech.wake_words
        sensitivity = sensitivity or self.config.speech.wake_sensitivity

        async for event in self._backend.detect_wake(audio_source, wake_words, sensitivity):
            # Publish event
            await self._event_bus.publish(
                Event(
                    topic="audio.wake",
                    data={
                        "wake_word": event.wake_word,
                        "confidence": event.confidence,
                    },
                    source="speech",
                )
            )
            yield event

    def get_status(self) -> dict:
        """Get speech status."""
        if not self._backend:
            return {"stt_available": False, "tts_available": False}
        return self._backend.get_status()


def main() -> None:
    """Entry point for Speech service."""
    service = SpeechService()
    service.run()


if __name__ == "__main__":
    main()


