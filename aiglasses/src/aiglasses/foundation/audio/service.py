"""Audio service implementation."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from dataclasses import dataclass, field
from typing import AsyncIterator

from grpc import aio as grpc_aio

from aiglasses.common import BaseService, get_logger
from aiglasses.common.events import Event, get_event_bus
from aiglasses.config import Config


@dataclass
class AudioDevice:
    """Audio device information."""

    address: str
    name: str
    type: str  # "bluetooth", "usb", "builtin"
    connected: bool = False
    paired: bool = False
    supported_profiles: list[str] = field(default_factory=list)
    battery_level: int = -1
    signal_strength: int = -1


@dataclass
class AudioChunk:
    """Audio data chunk."""

    data: bytes
    timestamp_ms: int
    sample_rate: int
    channels: int
    format: str
    sequence: int


class AudioBackend:
    """Abstract audio backend."""

    async def setup(self) -> None:
        """Setup audio backend."""
        pass

    async def teardown(self) -> None:
        """Teardown audio backend."""
        pass

    async def list_devices(self) -> list[AudioDevice]:
        """List available audio devices."""
        raise NotImplementedError

    async def connect(self, address: str, profile: str = "auto") -> AudioDevice | None:
        """Connect to a device."""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Disconnect from current device."""
        raise NotImplementedError

    async def open_mic(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        format: str = "pcm_s16le",
        chunk_size_ms: int = 100,
    ) -> AsyncIterator[AudioChunk]:
        """Open microphone stream."""
        raise NotImplementedError

    async def play_audio(self, chunks: AsyncIterator[AudioChunk]) -> None:
        """Play audio chunks."""
        raise NotImplementedError

    async def play_bytes(
        self,
        audio_data: bytes,
        format: str = "wav",
        sample_rate: int = 22050,
    ) -> None:
        """Play audio from bytes."""
        raise NotImplementedError

    def get_status(self) -> dict:
        """Get audio status."""
        raise NotImplementedError


class MockAudioBackend(AudioBackend):
    """Mock audio backend for testing."""

    def __init__(self) -> None:
        self._connected_device: AudioDevice | None = None
        self._mic_active = False
        self._playback_active = False
        self._volume = 80

    async def setup(self) -> None:
        """Setup mock audio."""
        pass

    async def teardown(self) -> None:
        """Teardown mock audio."""
        self._connected_device = None

    async def list_devices(self) -> list[AudioDevice]:
        """List mock devices."""
        return [
            AudioDevice(
                address="AA:BB:CC:DD:EE:FF",
                name="Mock AirPods",
                type="bluetooth",
                connected=self._connected_device is not None,
                paired=True,
                supported_profiles=["hfp", "a2dp"],
                battery_level=85,
                signal_strength=-45,
            ),
            AudioDevice(
                address="builtin",
                name="Built-in Audio",
                type="builtin",
                connected=True,
                paired=True,
                supported_profiles=["pcm"],
            ),
        ]

    async def connect(self, address: str, profile: str = "auto") -> AudioDevice | None:
        """Connect to mock device."""
        devices = await self.list_devices()
        for device in devices:
            if device.address == address:
                device.connected = True
                self._connected_device = device
                return device
        return None

    async def disconnect(self) -> None:
        """Disconnect mock device."""
        if self._connected_device:
            self._connected_device.connected = False
            self._connected_device = None

    async def open_mic(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        format: str = "pcm_s16le",
        chunk_size_ms: int = 100,
    ) -> AsyncIterator[AudioChunk]:
        """Open mock microphone stream."""
        self._mic_active = True
        sequence = 0
        chunk_samples = int(sample_rate * chunk_size_ms / 1000)
        bytes_per_sample = 2  # 16-bit

        try:
            while self._mic_active:
                # Generate silence (zeros)
                chunk_data = bytes(chunk_samples * channels * bytes_per_sample)

                yield AudioChunk(
                    data=chunk_data,
                    timestamp_ms=int(time.time() * 1000),
                    sample_rate=sample_rate,
                    channels=channels,
                    format=format,
                    sequence=sequence,
                )
                sequence += 1

                await asyncio.sleep(chunk_size_ms / 1000)
        finally:
            self._mic_active = False

    async def play_audio(self, chunks: AsyncIterator[AudioChunk]) -> None:
        """Play mock audio chunks."""
        self._playback_active = True
        try:
            async for chunk in chunks:
                # Mock playback - just consume the chunks
                await asyncio.sleep(len(chunk.data) / (chunk.sample_rate * 2))
        finally:
            self._playback_active = False

    async def play_bytes(
        self,
        audio_data: bytes,
        format: str = "wav",
        sample_rate: int = 22050,
    ) -> None:
        """Play mock audio bytes."""
        self._playback_active = True
        try:
            # Mock playback duration based on data size
            duration = len(audio_data) / (sample_rate * 2)
            await asyncio.sleep(min(duration, 0.1))  # Cap at 100ms for mock
        finally:
            self._playback_active = False

    def get_status(self) -> dict:
        """Get mock audio status."""
        return {
            "available": True,
            "current_device": {
                "address": self._connected_device.address if self._connected_device else None,
                "name": self._connected_device.name if self._connected_device else None,
            },
            "active_profile": "hfp" if self._connected_device else "none",
            "mic_active": self._mic_active,
            "playback_active": self._playback_active,
            "volume": self._volume,
        }


class PipeWireAudioBackend(AudioBackend):
    """PipeWire/PulseAudio audio backend for real hardware."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._connected_device: AudioDevice | None = None
        self._mic_active = False
        self._playback_active = False
        self._volume = 80
        self.logger = get_logger("pipewire_audio_backend")

    async def setup(self) -> None:
        """Setup PipeWire audio."""
        # PipeWire setup would happen here
        pass

    async def teardown(self) -> None:
        """Teardown PipeWire audio."""
        pass

    async def list_devices(self) -> list[AudioDevice]:
        """List audio devices via PipeWire/BlueZ."""
        devices = []

        try:
            # Use pactl to list sinks and sources
            proc = await asyncio.create_subprocess_exec(
                "pactl", "list", "sinks", "short",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            for line in stdout.decode().strip().split("\n"):
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        name = parts[1]
                        device_type = "bluetooth" if "bluez" in name.lower() else "builtin"
                        devices.append(
                            AudioDevice(
                                address=name,
                                name=name,
                                type=device_type,
                                connected=True,
                                paired=True,
                                supported_profiles=["pcm"],
                            )
                        )

        except Exception as e:
            self.logger.warning("list_devices_failed", error=str(e))

        return devices

    async def connect(self, address: str, profile: str = "auto") -> AudioDevice | None:
        """Connect to Bluetooth device."""
        try:
            # Use bluetoothctl to connect
            proc = await asyncio.create_subprocess_exec(
                "bluetoothctl", "connect", address,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode == 0:
                self._connected_device = AudioDevice(
                    address=address,
                    name=address,
                    type="bluetooth",
                    connected=True,
                    paired=True,
                    supported_profiles=["hfp", "a2dp"],
                )
                return self._connected_device
        except Exception as e:
            self.logger.exception("bluetooth_connect_failed", error=str(e))

        return None

    async def disconnect(self) -> None:
        """Disconnect Bluetooth device."""
        if self._connected_device:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "bluetoothctl", "disconnect", self._connected_device.address,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            except Exception as e:
                self.logger.warning("bluetooth_disconnect_failed", error=str(e))
            finally:
                self._connected_device = None

    async def open_mic(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        format: str = "pcm_s16le",
        chunk_size_ms: int = 100,
    ) -> AsyncIterator[AudioChunk]:
        """Open microphone stream via PipeWire."""
        try:
            import sounddevice as sd
        except ImportError:
            self.logger.error("sounddevice_not_available")
            return

        self._mic_active = True
        sequence = 0
        chunk_samples = int(sample_rate * chunk_size_ms / 1000)

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                blocksize=chunk_samples,
            ) as stream:
                while self._mic_active:
                    data, _ = stream.read(chunk_samples)
                    yield AudioChunk(
                        data=data.tobytes(),
                        timestamp_ms=int(time.time() * 1000),
                        sample_rate=sample_rate,
                        channels=channels,
                        format=format,
                        sequence=sequence,
                    )
                    sequence += 1
        finally:
            self._mic_active = False

    async def play_audio(self, chunks: AsyncIterator[AudioChunk]) -> None:
        """Play audio via PipeWire."""
        try:
            import sounddevice as sd
        except ImportError:
            self.logger.error("sounddevice_not_available")
            return

        self._playback_active = True
        try:
            async for chunk in chunks:
                import numpy as np
                audio = np.frombuffer(chunk.data, dtype=np.int16)
                sd.play(audio, chunk.sample_rate)
                sd.wait()
        finally:
            self._playback_active = False

    async def play_bytes(
        self,
        audio_data: bytes,
        format: str = "wav",
        sample_rate: int = 22050,
    ) -> None:
        """Play audio bytes via PipeWire."""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            self.logger.error("sounddevice_not_available")
            return

        self._playback_active = True
        try:
            if format == "wav":
                with io.BytesIO(audio_data) as f:
                    with wave.open(f, "rb") as wf:
                        sample_rate = wf.getframerate()
                        data = wf.readframes(wf.getnframes())
                        audio = np.frombuffer(data, dtype=np.int16)
            else:
                audio = np.frombuffer(audio_data, dtype=np.int16)

            sd.play(audio, sample_rate)
            sd.wait()
        finally:
            self._playback_active = False

    def get_status(self) -> dict:
        """Get PipeWire audio status."""
        return {
            "available": True,
            "current_device": {
                "address": self._connected_device.address if self._connected_device else None,
                "name": self._connected_device.name if self._connected_device else None,
            },
            "active_profile": "hfp" if self._connected_device else "none",
            "mic_active": self._mic_active,
            "playback_active": self._playback_active,
            "volume": self._volume,
        }


class AudioService(BaseService):
    """Audio service.

    Responsibilities:
    - Bluetooth device management (AirPods)
    - Microphone capture
    - Audio playback
    """

    def __init__(self, config: Config | None = None, mock_mode: bool = False) -> None:
        super().__init__("audio", config, mock_mode)
        self._backend: AudioBackend | None = None
        self._event_bus = get_event_bus()

    @property
    def port(self) -> int:
        return self.config.ports.audio

    async def setup(self) -> None:
        """Setup Audio service."""
        self.logger.info("audio_service_setup", mock_mode=self.mock_mode)

        if self.mock_mode:
            self._backend = MockAudioBackend()
        else:
            self._backend = PipeWireAudioBackend(self.config)

        await self._backend.setup()

        # Auto-connect to configured device
        if self.config.audio.bluetooth_device:
            await self.connect(self.config.audio.bluetooth_device)

    async def teardown(self) -> None:
        """Teardown Audio service."""
        self.logger.info("audio_service_teardown")

        if self._backend:
            await self._backend.teardown()

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services."""
        from aiglasses.foundation.audio.grpc_servicer import AudioServicer, add_servicer

        servicer = AudioServicer(self)
        add_servicer(server, servicer)

    async def list_devices(self) -> list[AudioDevice]:
        """List available audio devices."""
        if not self._backend:
            return []
        return await self._backend.list_devices()

    async def connect(self, address: str, profile: str = "auto") -> AudioDevice | None:
        """Connect to an audio device.

        Args:
            address: Device address (Bluetooth MAC or device name).
            profile: Audio profile ("hfp", "a2dp", "auto").

        Returns:
            Connected device or None if failed.
        """
        if not self._backend:
            return None

        device = await self._backend.connect(address, profile)
        if device:
            await self._event_bus.publish(
                Event(
                    topic="audio.connected",
                    data={
                        "address": device.address,
                        "name": device.name,
                        "type": device.type,
                    },
                    source="audio",
                )
            )
        return device

    async def disconnect(self) -> None:
        """Disconnect from current device."""
        if self._backend:
            await self._backend.disconnect()
            await self._event_bus.publish(
                Event(
                    topic="audio.disconnected",
                    data={},
                    source="audio",
                )
            )

    async def open_mic(
        self,
        sample_rate: int | None = None,
        channels: int | None = None,
        chunk_size_ms: int | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Open microphone stream.

        Args:
            sample_rate: Sample rate in Hz.
            channels: Number of channels.
            chunk_size_ms: Chunk size in milliseconds.

        Yields:
            Audio chunks.
        """
        if not self._backend:
            return

        sample_rate = sample_rate or self.config.audio.sample_rate
        channels = channels or self.config.audio.channels
        chunk_size_ms = chunk_size_ms or self.config.audio.chunk_size_ms

        async for chunk in self._backend.open_mic(sample_rate, channels, "pcm_s16le", chunk_size_ms):
            yield chunk

    async def play_audio(self, chunks: AsyncIterator[AudioChunk]) -> None:
        """Play audio from chunks.

        Args:
            chunks: Audio chunk iterator.
        """
        if self._backend:
            await self._backend.play_audio(chunks)

    async def play_bytes(
        self,
        audio_data: bytes,
        format: str = "wav",
        sample_rate: int = 22050,
    ) -> None:
        """Play audio from bytes.

        Args:
            audio_data: Audio data.
            format: Audio format ("wav", "mp3", "opus").
            sample_rate: Sample rate for raw audio.
        """
        if self._backend:
            await self._backend.play_bytes(audio_data, format, sample_rate)

    def get_status(self) -> dict:
        """Get audio status."""
        if not self._backend:
            return {"available": False, "error_message": "Backend not initialized"}
        return self._backend.get_status()


def main() -> None:
    """Entry point for Audio service."""
    service = AudioService()
    service.run()


if __name__ == "__main__":
    main()


