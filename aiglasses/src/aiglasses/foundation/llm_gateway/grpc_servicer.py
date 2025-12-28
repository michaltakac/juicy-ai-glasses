"""gRPC servicer for LLM Gateway service."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

import grpc
from grpc import aio as grpc_aio

from aiglasses.foundation.llm_gateway.service import Message

if TYPE_CHECKING:
    from aiglasses.foundation.llm_gateway.service import LLMGatewayService


class LLMGatewayServicer:
    """gRPC servicer for LLM Gateway service."""

    def __init__(self, service: LLMGatewayService) -> None:
        self.service = service

    async def GetHealth(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get service health."""
        status = self.service.get_status()
        return {
            "service_name": "llm_gateway",
            "status": "healthy" if status.get("available") else "unhealthy",
            "message": "",
        }

    async def GetStatus(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Get gateway status."""
        return self.service.get_status()

    async def Chat(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[dict]:
        """Chat with LLM (streaming)."""
        messages = [
            Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                name=m.get("name"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in request.get("messages", [])
        ]

        model = request.get("model")
        provider = request.get("provider")
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 1024)
        tools = request.get("tools")
        stream = request.get("stream", True)

        async for response in self.service.chat(
            messages, model, provider, temperature, max_tokens, tools, stream
        ):
            yield {
                "content": response.content,
                "done": response.done,
                "finish_reason": response.finish_reason,
                "tool_calls": response.tool_calls,
                "usage": response.usage,
                "latency_ms": response.latency_ms,
            }

    async def ChatComplete(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Chat with LLM (non-streaming)."""
        messages = [
            Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
            )
            for m in request.get("messages", [])
        ]

        model = request.get("model")
        provider = request.get("provider")
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 1024)
        tools = request.get("tools")

        # Collect all responses
        full_content = ""
        last_response = None
        async for response in self.service.chat(
            messages, model, provider, temperature, max_tokens, tools, stream=False
        ):
            full_content += response.content
            last_response = response

        return {
            "message": {
                "role": "assistant",
                "content": full_content,
            },
            "finish_reason": last_response.finish_reason if last_response else "stop",
            "usage": last_response.usage if last_response else {},
            "latency_ms": last_response.latency_ms if last_response else 0,
        }

    async def ListProviders(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """List available providers."""
        status = self.service.get_status()
        return {"providers": status.get("providers", [])}

    async def TestProvider(
        self,
        request: dict,
        context: grpc.aio.ServicerContext,
    ) -> dict:
        """Test provider connectivity."""
        provider_id = request.get("provider_id", "")
        success, latency_ms, message = await self.service.test_provider(provider_id)
        return {
            "success": success,
            "latency_ms": latency_ms,
            "message": message,
        }


def add_servicer(server: grpc_aio.Server, servicer: LLMGatewayServicer) -> None:
    """Add LLM Gateway servicer to gRPC server."""
    from aiglasses.common.grpc_utils import GenericServiceHandler

    handler = GenericServiceHandler(
        servicer,
        service_name="aiglasses.llm_gateway.LLMGatewayService",
        methods={
            "GetHealth": ("unary", "unary"),
            "GetStatus": ("unary", "unary"),
            "Chat": ("unary", "stream"),
            "ChatComplete": ("unary", "unary"),
            "ListProviders": ("unary", "unary"),
            "TestProvider": ("unary", "unary"),
        },
    )
    server.add_generic_rpc_handlers((handler,))


