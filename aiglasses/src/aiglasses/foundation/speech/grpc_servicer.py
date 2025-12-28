"""gRPC servicer for Speech service."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, AsyncIterator

import grpc
from grpc import aio as grpc_aio

if TYPE_CHECKING:
    from aiglasses.foundation.speech.service import SpeechService


class SpeechServicer:
    """gRPC servicer for Speech service."""

    def __init__(self, service: SpeechService) -> None:
        self.service = service

    async def GetHealth(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get service health."""
        status = self.service.get_status()
        return {
            "service_name": "speech",
            "status": "healthy" if status.get("stt_available") else "degraded",
            "message": "",
        }

    async def GetStatus(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get speech status."""
        return self.service.get_status()

    async def TranscribeBytes(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Transcribe audio bytes."""
        audio_data = base64.b64decode(request.get("audio_data", ""))
        format = request.get("format", "wav")
        language = request.get("language", "en")

        result = await self.service.transcribe_bytes(audio_data, format, language)

        return {
            "text": result.text,
            "is_final": result.is_final,
            "confidence": result.confidence,
            "language": result.language,
            "latency_ms": result.latency_ms,
        }

    async def SynthesizeComplete(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Synthesize text to speech."""
        text = request.get("text", "")
        voice = request.get("voice")
        speed = request.get("speed", 1.0)
        format = request.get("format", "wav")

        result = await self.service.synthesize_complete(text, voice, speed, format)

        return {
            "audio_data": base64.b64encode(result.audio_data).decode("ascii"),
            "format": result.format,
            "sample_rate": result.sample_rate,
            "duration_ms": result.duration_ms,
            "latency_ms": result.latency_ms,
        }

    async def Synthesize(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[dict]:
        """Synthesize text to speech (streaming)."""
        text = request.get("text", "")
        voice = request.get("voice")
        speed = request.get("speed", 1.0)
        format = request.get("format", "wav")

        sequence = 0
        async for chunk in self.service.synthesize(text, voice, speed, format):
            yield {
                "audio_data": base64.b64encode(chunk).decode("ascii"),
                "is_final": False,
                "sequence": sequence,
            }
            sequence += 1

        yield {"audio_data": "", "is_final": True, "sequence": sequence}


def add_servicer(server: grpc_aio.Server, servicer: SpeechServicer) -> None:
    """Add Speech servicer to gRPC server."""
    from aiglasses.common.grpc_utils import GenericServiceHandler

    handler = GenericServiceHandler(
        servicer,
        service_name="aiglasses.speech.SpeechService",
        methods={
            "GetHealth": ("unary", "unary"),
            "GetStatus": ("unary", "unary"),
            "TranscribeBytes": ("unary", "unary"),
            "SynthesizeComplete": ("unary", "unary"),
            "Synthesize": ("unary", "stream"),
        },
    )
    server.add_generic_rpc_handlers((handler,))


