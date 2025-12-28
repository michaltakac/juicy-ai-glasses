"""LLM Gateway service implementation."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Any

import httpx
from grpc import aio as grpc_aio

from aiglasses.common import BaseService, get_logger
from aiglasses.common.events import Event, get_event_bus
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
class ChatResponse:
    """Chat response."""

    content: str
    done: bool
    finish_reason: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict | None = None
    latency_ms: int = 0


@dataclass
class ProviderStatus:
    """LLM provider status."""

    id: str
    name: str
    type: str
    available: bool
    configured: bool
    latency_ms: int = 0
    error: str | None = None


class LLMProvider:
    """Abstract LLM provider."""

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[ChatResponse]:
        """Chat with the LLM."""
        raise NotImplementedError

    async def test(self) -> tuple[bool, int, str]:
        """Test provider connectivity.

        Returns:
            Tuple of (success, latency_ms, message).
        """
        raise NotImplementedError

    def get_status(self) -> ProviderStatus:
        """Get provider status."""
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self) -> None:
        self._available = True

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[ChatResponse]:
        """Mock chat response."""
        # Generate a mock response based on the last user message
        user_messages = [m for m in messages if m.role == "user"]
        if user_messages:
            last_message = user_messages[-1].content.lower()
        else:
            last_message = ""

        response_text = "I can see a person working at a desk with a laptop. There appears to be a coffee cup nearby. The scene looks like a typical home office setup."

        if "what" in last_message and "seeing" in last_message:
            response_text = "Based on the image, I can see a person sitting at a desk with a laptop computer. There's also a coffee cup visible in the scene. The lighting suggests this is an indoor environment, likely a home office or workspace."

        if stream:
            # Stream response word by word
            words = response_text.split()
            for i, word in enumerate(words):
                await asyncio.sleep(0.02)  # Simulate streaming delay
                yield ChatResponse(
                    content=word + (" " if i < len(words) - 1 else ""),
                    done=i == len(words) - 1,
                    finish_reason="stop" if i == len(words) - 1 else None,
                    usage={"prompt_tokens": 100, "completion_tokens": len(words), "total_tokens": 100 + len(words)} if i == len(words) - 1 else None,
                    latency_ms=50 * (i + 1),
                )
        else:
            await asyncio.sleep(0.1)
            yield ChatResponse(
                content=response_text,
                done=True,
                finish_reason="stop",
                usage={"prompt_tokens": 100, "completion_tokens": len(response_text.split()), "total_tokens": 100 + len(response_text.split())},
                latency_ms=100,
            )

    async def test(self) -> tuple[bool, int, str]:
        """Test mock provider."""
        await asyncio.sleep(0.01)
        return True, 10, "Mock provider ready"

    def get_status(self) -> ProviderStatus:
        """Get mock provider status."""
        return ProviderStatus(
            id="mock",
            name="Mock Provider",
            type="mock",
            available=self._available,
            configured=True,
            latency_ms=10,
        )


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible LLM provider (works with OpenAI, Ollama, LAN servers)."""

    def __init__(
        self,
        provider_id: str,
        name: str,
        endpoint: str,
        api_key: str | None = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self.provider_id = provider_id
        self.name = name
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self._available = False
        self._last_latency_ms = 0
        self._last_error: str | None = None
        self.logger = get_logger(f"llm_provider_{provider_id}")

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[ChatResponse]:
        """Chat using OpenAI-compatible API."""
        model = model or self.default_model
        start_time = time.time()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if stream:
                    async with client.stream(
                        "POST",
                        f"{self.endpoint}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    yield ChatResponse(
                                        content="",
                                        done=True,
                                        finish_reason="stop",
                                        latency_ms=int((time.time() - start_time) * 1000),
                                    )
                                    break

                                try:
                                    chunk = json.loads(data)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    content = delta.get("content", "")
                                    finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

                                    if content or finish_reason:
                                        yield ChatResponse(
                                            content=content,
                                            done=finish_reason is not None,
                                            finish_reason=finish_reason,
                                            latency_ms=int((time.time() - start_time) * 1000),
                                        )
                                except json.JSONDecodeError:
                                    continue
                else:
                    response = await client.post(
                        f"{self.endpoint}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    result = response.json()

                    choice = result.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    usage = result.get("usage", {})

                    yield ChatResponse(
                        content=message.get("content", ""),
                        done=True,
                        finish_reason=choice.get("finish_reason", "stop"),
                        tool_calls=message.get("tool_calls"),
                        usage={
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                        latency_ms=int((time.time() - start_time) * 1000),
                    )

            self._available = True
            self._last_error = None

        except Exception as e:
            self._available = False
            self._last_error = str(e)
            self.logger.exception("chat_failed", error=str(e))
            raise

    async def test(self) -> tuple[bool, int, str]:
        """Test provider connectivity."""
        start_time = time.time()

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try models endpoint first
                response = await client.get(
                    f"{self.endpoint}/models",
                    headers=headers,
                )
                response.raise_for_status()

            latency_ms = int((time.time() - start_time) * 1000)
            self._available = True
            self._last_latency_ms = latency_ms
            self._last_error = None
            return True, latency_ms, "Connected successfully"

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            self._available = False
            self._last_error = str(e)
            return False, latency_ms, str(e)

    def get_status(self) -> ProviderStatus:
        """Get provider status."""
        return ProviderStatus(
            id=self.provider_id,
            name=self.name,
            type="openai_compatible",
            available=self._available,
            configured=bool(self.endpoint),
            latency_ms=self._last_latency_ms,
            error=self._last_error,
        )


class ClaudeCodeProvider(LLMProvider):
    """Claude Code SDK provider - uses local Claude Code installation with Pro subscription."""

    def __init__(
        self,
        default_model: str = "sonnet",  # sonnet, opus, haiku
    ) -> None:
        self.default_model = default_model
        self._available = False
        self._last_latency_ms = 0
        self._last_error: str | None = None
        self.logger = get_logger("claude_code_provider")

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[ChatResponse]:
        """Chat using Claude Code SDK with ClaudeSDKClient."""
        from claude_agent_sdk import (
            ClaudeSDKClient,
            ClaudeAgentOptions,
            AssistantMessage,
            TextBlock,
            ResultMessage,
        )

        start_time = time.time()

        # Build prompt from messages
        system_prompt = None
        user_prompt_parts = []

        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            elif m.role == "user":
                user_prompt_parts.append(m.content)
            elif m.role == "assistant":
                user_prompt_parts.append(f"[Previous assistant response: {m.content}]")

        prompt = "\n".join(user_prompt_parts)

        # Map model names
        model_map = {
            "sonnet": "sonnet",
            "claude-sonnet-4-20250514": "sonnet",
            "opus": "opus",
            "haiku": "haiku",
        }
        sdk_model = model_map.get(model or self.default_model, "sonnet")

        options = ClaudeAgentOptions(
            model=sdk_model,
            system_prompt=system_prompt,
            # No tools for simple chat - just text responses
            allowed_tools=[],
            disallowed_tools=["Bash", "Write", "Edit", "Read"],  # Safety: no file/command access
        )

        try:
            full_content = ""
            usage_info = None

            # Use context manager for proper cleanup
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                full_content += block.text
                                if stream:
                                    yield ChatResponse(
                                        content=block.text,
                                        done=False,
                                        latency_ms=int((time.time() - start_time) * 1000),
                                    )

                    elif isinstance(message, ResultMessage):
                        usage_info = {
                            "total_cost_usd": message.total_cost_usd,
                            "duration_ms": message.duration_ms,
                        }
                        if message.usage:
                            usage_info.update(message.usage)

            # Final response
            yield ChatResponse(
                content="" if stream else full_content,
                done=True,
                finish_reason="stop",
                usage=usage_info,
                latency_ms=int((time.time() - start_time) * 1000),
            )

            self._available = True
            self._last_error = None

        except Exception as e:
            self._available = False
            self._last_error = str(e)
            self.logger.exception("claude_code_chat_failed", error=str(e))
            raise

    async def test(self) -> tuple[bool, int, str]:
        """Test Claude Code SDK connectivity."""
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage

        start_time = time.time()

        try:
            options = ClaudeAgentOptions(
                model="sonnet",
                allowed_tools=[],
                disallowed_tools=["Bash", "Write", "Edit", "Read"],
            )

            async with ClaudeSDKClient(options=options) as client:
                await client.query("Say 'ok' and nothing else.")
                async for message in client.receive_response():
                    if isinstance(message, ResultMessage):
                        break

            latency_ms = int((time.time() - start_time) * 1000)
            self._available = True
            self._last_latency_ms = latency_ms
            self._last_error = None
            return True, latency_ms, "Claude Code SDK connected"

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            self._available = False
            self._last_error = str(e)
            return False, latency_ms, str(e)

    def get_status(self) -> ProviderStatus:
        """Get Claude Code provider status."""
        return ProviderStatus(
            id="claude_code",
            name="Claude Code SDK",
            type="claude_code",
            available=self._available,
            configured=True,  # Always configured if Claude Code is installed
            latency_ms=self._last_latency_ms,
            error=self._last_error,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic/Claude LLM provider using the official SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.api_key = api_key
        self.default_model = default_model
        self._available = False
        self._last_latency_ms = 0
        self._last_error: str | None = None
        self._client = None
        self.logger = get_logger("anthropic_provider")

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                self.logger.error("failed_to_create_client", error=str(e))
                raise
        return self._client

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[ChatResponse]:
        """Chat using Anthropic API."""
        model = model or self.default_model
        start_time = time.time()

        # Separate system message from other messages
        system_content = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_content = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        try:
            client = self._get_client()

            if stream:
                # Use streaming
                with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_content if system_content else None,
                    messages=chat_messages,
                    temperature=temperature,
                ) as stream_response:
                    for text in stream_response.text_stream:
                        yield ChatResponse(
                            content=text,
                            done=False,
                            latency_ms=int((time.time() - start_time) * 1000),
                        )

                # Final response with usage
                final_message = stream_response.get_final_message()
                yield ChatResponse(
                    content="",
                    done=True,
                    finish_reason=final_message.stop_reason or "stop",
                    usage={
                        "prompt_tokens": final_message.usage.input_tokens,
                        "completion_tokens": final_message.usage.output_tokens,
                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
                    },
                    latency_ms=int((time.time() - start_time) * 1000),
                )
            else:
                # Non-streaming
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_content if system_content else None,
                    messages=chat_messages,
                    temperature=temperature,
                )

                content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

                yield ChatResponse(
                    content=content,
                    done=True,
                    finish_reason=response.stop_reason or "stop",
                    usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    },
                    latency_ms=int((time.time() - start_time) * 1000),
                )

            self._available = True
            self._last_error = None

        except Exception as e:
            self._available = False
            self._last_error = str(e)
            self.logger.exception("anthropic_chat_failed", error=str(e))
            raise

    async def test(self) -> tuple[bool, int, str]:
        """Test Anthropic connectivity."""
        start_time = time.time()

        try:
            client = self._get_client()
            # Simple test - create a minimal message
            response = client.messages.create(
                model=self.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )

            latency_ms = int((time.time() - start_time) * 1000)
            self._available = True
            self._last_latency_ms = latency_ms
            self._last_error = None
            return True, latency_ms, f"Connected to {self.default_model}"

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            self._available = False
            self._last_error = str(e)
            return False, latency_ms, str(e)

    def get_status(self) -> ProviderStatus:
        """Get Anthropic provider status."""
        return ProviderStatus(
            id="anthropic",
            name="Anthropic Claude",
            type="anthropic",
            available=self._available,
            configured=bool(self.api_key),
            latency_ms=self._last_latency_ms,
            error=self._last_error,
        )


class LLMGatewayService(BaseService):
    """LLM Gateway service.

    Responsibilities:
    - Broker LLM requests to LAN/cloud providers
    - Handle streaming responses
    - Manage failover and retries
    """

    def __init__(self, config: Config | None = None, mock_mode: bool = False) -> None:
        super().__init__("llm_gateway", config, mock_mode)
        self._providers: dict[str, LLMProvider] = {}
        self._active_provider: str = ""
        self._event_bus = get_event_bus()
        self._total_requests = 0
        self._total_tokens = 0

    @property
    def port(self) -> int:
        return self.config.ports.llm_gateway

    async def setup(self) -> None:
        """Setup LLM Gateway service."""
        self.logger.info("llm_gateway_setup", mock_mode=self.mock_mode)

        if self.mock_mode:
            self._providers["mock"] = MockLLMProvider()
            self._active_provider = "mock"
        else:
            # Setup configured providers
            llm_config = self.config.llm

            # LAN provider
            if llm_config.lan_endpoint:
                self._providers["lan"] = OpenAICompatibleProvider(
                    "lan",
                    "LAN Server",
                    llm_config.lan_endpoint,
                    llm_config.api_key,
                    llm_config.default_model,
                )

            # OpenAI provider
            if llm_config.api_key:
                self._providers["openai"] = OpenAICompatibleProvider(
                    "openai",
                    "OpenAI",
                    llm_config.openai_endpoint,
                    llm_config.api_key,
                    "gpt-4o-mini",
                )

            # Ollama provider
            if llm_config.ollama_endpoint:
                self._providers["ollama"] = OpenAICompatibleProvider(
                    "ollama",
                    "Ollama",
                    llm_config.ollama_endpoint,
                    None,
                    "llama3",
                )

            # Anthropic provider
            if llm_config.api_key or llm_config.provider == "anthropic":
                import os
                anthropic_key = llm_config.api_key or os.environ.get("ANTHROPIC_API_KEY")
                if anthropic_key:
                    self._providers["anthropic"] = AnthropicProvider(
                        api_key=anthropic_key,
                        default_model="claude-sonnet-4-20250514",
                    )

            # Claude Code SDK provider (uses local Claude Code installation)
            # This uses the Pro subscription authentication
            self._providers["claude_code"] = ClaudeCodeProvider(
                default_model="sonnet",
            )

            # Set active provider - prefer claude_code if available
            self._active_provider = llm_config.provider
            if self._active_provider not in self._providers:
                # Default to claude_code if configured provider not available
                if "claude_code" in self._providers:
                    self._active_provider = "claude_code"
            if self._active_provider not in self._providers and self._providers:
                self._active_provider = next(iter(self._providers))

    async def teardown(self) -> None:
        """Teardown LLM Gateway service."""
        self.logger.info("llm_gateway_teardown")

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services."""
        from aiglasses.foundation.llm_gateway.grpc_servicer import (
            LLMGatewayServicer,
            add_servicer,
        )

        servicer = LLMGatewayServicer(self)
        add_servicer(server, servicer)

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[ChatResponse]:
        """Chat with an LLM.

        Args:
            messages: Chat messages.
            model: Model to use.
            provider: Provider to use.
            temperature: Temperature setting.
            max_tokens: Maximum tokens.
            tools: Available tools.
            stream: Whether to stream.

        Yields:
            Chat responses.
        """
        provider_id = provider or self._active_provider
        llm_provider = self._providers.get(provider_id)

        if not llm_provider:
            raise ValueError(f"Provider {provider_id} not found")

        temperature = temperature or self.config.llm.default_temperature
        max_tokens = max_tokens or self.config.llm.default_max_tokens

        self._total_requests += 1

        try:
            async for response in llm_provider.chat(
                messages, model, temperature, max_tokens, tools, stream
            ):
                if response.usage:
                    self._total_tokens += response.usage.get("total_tokens", 0)
                yield response

        except Exception as e:
            # Try fallback provider
            fallback_id = self.config.llm.fallback_provider
            if fallback_id and fallback_id != provider_id:
                fallback = self._providers.get(fallback_id)
                if fallback:
                    self.logger.warning(
                        "using_fallback_provider",
                        primary=provider_id,
                        fallback=fallback_id,
                        error=str(e),
                    )
                    async for response in fallback.chat(
                        messages, model, temperature, max_tokens, tools, stream
                    ):
                        yield response
                    return

            raise

    async def test_provider(self, provider_id: str) -> tuple[bool, int, str]:
        """Test a provider's connectivity.

        Args:
            provider_id: Provider ID.

        Returns:
            Tuple of (success, latency_ms, message).
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return False, 0, f"Provider {provider_id} not found"
        return await provider.test()

    def get_status(self) -> dict:
        """Get gateway status."""
        providers = [p.get_status() for p in self._providers.values()]

        return {
            "available": bool(self._providers),
            "active_provider": self._active_provider,
            "providers": [
                {
                    "id": p.id,
                    "name": p.name,
                    "type": p.type,
                    "available": p.available,
                    "configured": p.configured,
                    "latency_ms": p.latency_ms,
                    "error": p.error,
                }
                for p in providers
            ],
            "metrics": {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
            },
        }


def main() -> None:
    """Entry point for LLM Gateway service."""
    service = LLMGatewayService()
    service.run()


if __name__ == "__main__":
    main()

