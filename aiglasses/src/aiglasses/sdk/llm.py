"""SDK LLM API - language model interactions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Any

from aiglasses.common.logging import get_logger
from aiglasses.common.grpc_utils import GrpcClient
from aiglasses.config import Config


@dataclass
class Message:
    """Chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass
class Tool:
    """Tool definition for function calling."""

    name: str
    description: str
    parameters: dict  # JSON Schema


@dataclass
class ToolCall:
    """Tool call from LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatChunk:
    """Streaming chat response chunk."""

    content: str
    done: bool
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None
    latency_ms: int = 0


@dataclass
class ChatResponse:
    """Complete chat response."""

    content: str
    finish_reason: str
    tool_calls: list[ToolCall] | None = None
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0


class LLMAPI:
    """LLM API for language model interactions.

    Provides methods for:
    - Chat completions (streaming and non-streaming)
    - Tool/function calling
    - Multiple provider support
    """

    def __init__(self, config: Config, mock_mode: bool = False) -> None:
        """Initialize LLM API.

        Args:
            config: Configuration.
            mock_mode: Run in mock mode.
        """
        self.config = config
        self.mock_mode = mock_mode
        self.logger = get_logger("sdk.llm")

        self._client: GrpcClient | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to LLM Gateway service."""
        if self._connected:
            return

        if not self.mock_mode:
            self._client = GrpcClient("localhost", self.config.ports.llm_gateway)
            await self._client.connect()

        self._connected = True
        self.logger.debug("llm_api_connected")

    async def disconnect(self) -> None:
        """Disconnect from service."""
        if self._client:
            await self._client.close()
        self._connected = False

    async def chat(
        self,
        messages: list[dict | Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        provider: str | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Chat with an LLM (streaming).

        Args:
            messages: Chat messages. Can be dicts or Message objects.
            model: Model to use (default from config).
            temperature: Temperature setting (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            tools: Available tools for function calling.
            provider: Provider to use (default from config).

        Yields:
            Streaming chat response chunks.

        Example:
            messages = [
                {"role": "user", "content": "What do you see in this image?"}
            ]
            async for chunk in app.llm.chat(messages):
                print(chunk.content, end="", flush=True)
        """
        temperature = temperature or self.config.llm.default_temperature
        max_tokens = max_tokens or self.config.llm.default_max_tokens

        # Convert messages to dicts
        msg_dicts = []
        for msg in messages:
            if isinstance(msg, Message):
                msg_dicts.append({
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name,
                    "tool_calls": msg.tool_calls,
                    "tool_call_id": msg.tool_call_id,
                })
            else:
                msg_dicts.append(msg)

        if self.mock_mode:
            # Generate mock response
            response_text = self._generate_mock_response(msg_dicts)
            words = response_text.split()

            for i, word in enumerate(words):
                await asyncio.sleep(0.02)
                yield ChatChunk(
                    content=word + (" " if i < len(words) - 1 else ""),
                    done=i == len(words) - 1,
                    finish_reason="stop" if i == len(words) - 1 else None,
                    latency_ms=20 * (i + 1),
                )
        else:
            # Convert tools to dicts
            tool_dicts = None
            if tools:
                tool_dicts = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    }
                    for t in tools
                ]

            async for chunk_data in self._client.call_stream(
                "aiglasses.llm_gateway.LLMGatewayService",
                "Chat",
                {
                    "messages": msg_dicts,
                    "model": model,
                    "provider": provider,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools": tool_dicts,
                    "stream": True,
                },
            ):
                tool_calls = None
                if chunk_data.get("tool_calls"):
                    tool_calls = [
                        ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("name", ""),
                            arguments=tc.get("arguments", {}),
                        )
                        for tc in chunk_data["tool_calls"]
                    ]

                yield ChatChunk(
                    content=chunk_data.get("content", ""),
                    done=chunk_data.get("done", False),
                    finish_reason=chunk_data.get("finish_reason"),
                    tool_calls=tool_calls,
                    latency_ms=chunk_data.get("latency_ms", 0),
                )

    async def chat_complete(
        self,
        messages: list[dict | Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        provider: str | None = None,
    ) -> ChatResponse:
        """Chat with an LLM (non-streaming).

        Same arguments as chat(), but returns complete response.

        Returns:
            Complete chat response.

        Example:
            response = await app.llm.chat_complete([
                {"role": "user", "content": "Describe what you see."}
            ])
            print(response.content)
        """
        # Collect streaming response
        content = ""
        last_chunk = None

        async for chunk in self.chat(
            messages, model, temperature, max_tokens, tools, provider
        ):
            content += chunk.content
            last_chunk = chunk

        return ChatResponse(
            content=content,
            finish_reason=last_chunk.finish_reason if last_chunk else "stop",
            tool_calls=last_chunk.tool_calls if last_chunk else None,
            latency_ms=last_chunk.latency_ms if last_chunk else 0,
        )

    def _generate_mock_response(self, messages: list[dict]) -> str:
        """Generate a mock response based on messages."""
        # Find the last user message
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return "Hello! How can I help you?"

        last_message = user_messages[-1].get("content", "").lower()

        # Context-aware mock responses
        if "see" in last_message or "seeing" in last_message:
            return (
                "Based on the image, I can see a person sitting at a desk with "
                "a laptop computer. There's also what appears to be a coffee cup "
                "on the desk. The setting looks like a home office or workspace "
                "with natural lighting coming from a window."
            )
        elif "describe" in last_message:
            return (
                "I can describe what I observe in the scene. There appears to be "
                "an indoor environment with a person and some common office items. "
                "The lighting suggests daytime, and the overall atmosphere seems "
                "calm and focused."
            )
        elif "help" in last_message:
            return (
                "I'm here to help! I can describe what you're seeing, identify "
                "objects in your view, read text, and answer questions about "
                "your surroundings. Just ask me anything!"
            )
        else:
            return (
                "I understand. Let me help you with that. Based on what I can "
                "observe, I'll do my best to provide a helpful response."
            )

    async def ask(
        self,
        question: str,
        context: str | None = None,
        detections: list[dict] | None = None,
    ) -> str:
        """Simple helper to ask a question.

        This is a convenience method that builds a prompt and returns
        just the text response.

        Args:
            question: Question to ask.
            context: Optional context to include.
            detections: Optional object detections to include.

        Returns:
            Text response from LLM.

        Example:
            answer = await app.llm.ask(
                "What objects do you see?",
                detections=[{"label": "person"}, {"label": "laptop"}]
            )
            print(answer)
        """
        # Build prompt
        content = question
        if detections:
            labels = [d.get("label", "") for d in detections]
            content = f"Objects detected: {', '.join(labels)}. {question}"
        if context:
            content = f"{context}\n\n{content}"

        response = await self.chat_complete([
            {"role": "user", "content": content}
        ])

        return response.content


