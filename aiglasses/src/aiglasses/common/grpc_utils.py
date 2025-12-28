"""gRPC utilities for AI Glasses Platform."""

from __future__ import annotations

import json
from typing import Any, Callable

import grpc
from grpc import aio as grpc_aio


class GenericServiceHandler(grpc.GenericRpcHandler):
    """Generic gRPC service handler.

    Provides a simplified way to create gRPC services without proto compilation.
    Uses JSON serialization for messages.
    """

    def __init__(
        self,
        servicer: Any,
        service_name: str,
        methods: dict[str, tuple[str, str]],
    ) -> None:
        """Initialize the handler.

        Args:
            servicer: Service implementation with method handlers.
            service_name: Full service name (e.g., "package.ServiceName").
            methods: Dict mapping method name to (request_type, response_type).
                     Types are "unary" or "stream".
        """
        self.servicer = servicer
        self.service_name = service_name
        self.methods = methods

    def service(
        self,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | None:
        """Get the method handler for the given call details."""
        method = handler_call_details.method

        # Extract method name from full path (e.g., "/package.Service/Method")
        if "/" in method:
            parts = method.split("/")
            if len(parts) >= 3:
                service = parts[1]
                method_name = parts[2]
            else:
                return None
        else:
            return None

        if service != self.service_name:
            return None

        if method_name not in self.methods:
            return None

        request_type, response_type = self.methods[method_name]
        handler_fn = getattr(self.servicer, method_name, None)
        if handler_fn is None:
            return None

        if request_type == "unary" and response_type == "unary":
            return grpc.unary_unary_rpc_method_handler(
                self._wrap_unary_unary(handler_fn),
                request_deserializer=_json_deserialize,
                response_serializer=_json_serialize,
            )
        elif request_type == "unary" and response_type == "stream":
            return grpc.unary_stream_rpc_method_handler(
                self._wrap_unary_stream(handler_fn),
                request_deserializer=_json_deserialize,
                response_serializer=_json_serialize,
            )
        elif request_type == "stream" and response_type == "unary":
            return grpc.stream_unary_rpc_method_handler(
                self._wrap_stream_unary(handler_fn),
                request_deserializer=_json_deserialize,
                response_serializer=_json_serialize,
            )
        elif request_type == "stream" and response_type == "stream":
            return grpc.stream_stream_rpc_method_handler(
                self._wrap_stream_stream(handler_fn),
                request_deserializer=_json_deserialize,
                response_serializer=_json_serialize,
            )

        return None

    def _wrap_unary_unary(
        self,
        handler: Callable,
    ) -> Callable:
        """Wrap unary-unary handler."""

        async def wrapper(request: dict, context: grpc.aio.ServicerContext) -> dict:
            return await handler(request, context)

        return wrapper

    def _wrap_unary_stream(
        self,
        handler: Callable,
    ) -> Callable:
        """Wrap unary-stream handler."""

        async def wrapper(request: dict, context: grpc.aio.ServicerContext):
            async for response in handler(request, context):
                yield response

        return wrapper

    def _wrap_stream_unary(
        self,
        handler: Callable,
    ) -> Callable:
        """Wrap stream-unary handler."""

        async def wrapper(request_iterator, context: grpc.aio.ServicerContext) -> dict:
            return await handler(request_iterator, context)

        return wrapper

    def _wrap_stream_stream(
        self,
        handler: Callable,
    ) -> Callable:
        """Wrap stream-stream handler."""

        async def wrapper(request_iterator, context: grpc.aio.ServicerContext):
            async for response in handler(request_iterator, context):
                yield response

        return wrapper


def _json_serialize(data: dict) -> bytes:
    """Serialize dict to JSON bytes."""
    return json.dumps(data).encode("utf-8")


def _json_deserialize(data: bytes) -> dict:
    """Deserialize JSON bytes to dict."""
    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


class GrpcClient:
    """Generic gRPC client using JSON serialization."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the client.

        Args:
            host: Service host.
            port: Service port.
        """
        self.host = host
        self.port = port
        self._channel: grpc_aio.Channel | None = None

    async def connect(self) -> None:
        """Connect to the service."""
        self._channel = grpc_aio.insecure_channel(f"{self.host}:{self.port}")
        await self._channel.channel_ready()

    async def close(self) -> None:
        """Close the connection."""
        if self._channel:
            await self._channel.close()
            self._channel = None

    async def __aenter__(self) -> GrpcClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def call(
        self,
        service: str,
        method: str,
        request: dict,
        timeout: float | None = None,
    ) -> dict:
        """Make a unary-unary call.

        Args:
            service: Full service name.
            method: Method name.
            request: Request data.
            timeout: Call timeout.

        Returns:
            Response data.
        """
        if not self._channel:
            raise RuntimeError("Client not connected")

        call_path = f"/{service}/{method}"
        response = await self._channel.unary_unary(
            call_path,
            request_serializer=_json_serialize,
            response_deserializer=_json_deserialize,
        )(request, timeout=timeout)

        return response

    async def call_stream(
        self,
        service: str,
        method: str,
        request: dict,
        timeout: float | None = None,
    ):
        """Make a unary-stream call.

        Args:
            service: Full service name.
            method: Method name.
            request: Request data.
            timeout: Call timeout.

        Yields:
            Response data items.
        """
        if not self._channel:
            raise RuntimeError("Client not connected")

        call_path = f"/{service}/{method}"
        call = self._channel.unary_stream(
            call_path,
            request_serializer=_json_serialize,
            response_deserializer=_json_deserialize,
        )

        async for response in call(request, timeout=timeout):
            yield response


