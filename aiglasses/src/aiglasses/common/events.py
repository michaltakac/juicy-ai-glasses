"""Event bus for inter-service communication."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from aiglasses.common.logging import get_logger


@dataclass
class Event:
    """Event message."""

    topic: str
    data: dict[str, Any]
    source: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    correlation_id: str | None = None


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """In-process event bus for pub/sub messaging.

    For production, this would be backed by NATS, Redis pub/sub, or similar.
    This implementation provides a compatible API for local development.
    """

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._subscribers: dict[str, list[EventHandler]] = {}
        self._wildcard_subscribers: list[tuple[str, EventHandler]] = []
        self._history: list[Event] = []
        self._history_limit = 1000
        self._lock = asyncio.Lock()
        self.logger = get_logger("event_bus")

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish.
        """
        self.logger.debug(
            "publishing_event",
            topic=event.topic,
            event_id=event.event_id,
            source=event.source,
        )

        async with self._lock:
            # Store in history
            self._history.append(event)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]

        # Get exact topic subscribers
        handlers = self._subscribers.get(event.topic, []).copy()

        # Get wildcard subscribers
        for pattern, handler in self._wildcard_subscribers:
            if self._matches_pattern(event.topic, pattern):
                handlers.append(handler)

        # Dispatch to all handlers concurrently
        if handlers:
            await asyncio.gather(
                *[self._safe_dispatch(handler, event) for handler in handlers],
                return_exceptions=True,
            )

    async def _safe_dispatch(self, handler: EventHandler, event: Event) -> None:
        """Safely dispatch event to handler, catching exceptions."""
        try:
            await handler(event)
        except Exception as e:
            self.logger.exception(
                "event_handler_error",
                topic=event.topic,
                event_id=event.event_id,
                error=str(e),
            )

    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches a wildcard pattern.

        Supports:
        - * matches one segment
        - ** matches multiple segments

        Args:
            topic: Topic to check.
            pattern: Pattern to match against.

        Returns:
            True if topic matches pattern.
        """
        topic_parts = topic.split(".")
        pattern_parts = pattern.split(".")

        i, j = 0, 0
        while i < len(topic_parts) and j < len(pattern_parts):
            if pattern_parts[j] == "**":
                # ** matches rest of topic
                if j == len(pattern_parts) - 1:
                    return True
                # Try to match remaining pattern
                j += 1
                while i < len(topic_parts):
                    if self._matches_pattern(
                        ".".join(topic_parts[i:]),
                        ".".join(pattern_parts[j:]),
                    ):
                        return True
                    i += 1
                return False
            elif pattern_parts[j] == "*" or pattern_parts[j] == topic_parts[i]:
                i += 1
                j += 1
            else:
                return False

        return i == len(topic_parts) and j == len(pattern_parts)

    def subscribe(
        self,
        topic: str,
        handler: EventHandler | None = None,
    ) -> Callable[[EventHandler], EventHandler] | Callable[[], None]:
        """Subscribe to events on a topic.

        Can be used as a decorator or called directly.

        Args:
            topic: Topic to subscribe to. Supports wildcards (* and **).
            handler: Async function to handle events (optional for decorator use).

        Returns:
            Decorator (when handler is None) or unsubscribe function.
        """
        if handler is not None:
            # Direct call with handler
            return self._register_handler(topic, handler)

        # Decorator use
        def decorator(fn: EventHandler) -> EventHandler:
            self._register_handler(topic, fn)
            return fn

        return decorator

    def _register_handler(self, topic: str, handler: EventHandler) -> Callable[[], None]:
        """Register a handler and return unsubscribe function."""
        if "*" in topic:
            self._wildcard_subscribers.append((topic, handler))

            def unsubscribe() -> None:
                self._wildcard_subscribers.remove((topic, handler))

        else:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(handler)

            def unsubscribe() -> None:
                if topic in self._subscribers:
                    self._subscribers[topic].remove(handler)

        self.logger.debug("subscribed", topic=topic)
        return unsubscribe

    async def request(
        self,
        topic: str,
        data: dict[str, Any],
        source: str,
        timeout: float = 5.0,
    ) -> Event | None:
        """Publish a request and wait for response.

        Args:
            topic: Topic to publish to.
            data: Request data.
            source: Source identifier.
            timeout: Response timeout in seconds.

        Returns:
            Response event or None if timeout.
        """
        correlation_id = str(uuid.uuid4())
        response_topic = f"{topic}.response.{correlation_id}"
        response_event: Event | None = None
        response_received = asyncio.Event()

        async def response_handler(event: Event) -> None:
            nonlocal response_event
            response_event = event
            response_received.set()

        unsubscribe = self.subscribe(response_topic, response_handler)

        try:
            # Publish request
            request_event = Event(
                topic=topic,
                data=data,
                source=source,
                correlation_id=correlation_id,
            )
            await self.publish(request_event)

            # Wait for response
            try:
                await asyncio.wait_for(response_received.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning(
                    "request_timeout",
                    topic=topic,
                    correlation_id=correlation_id,
                )
                return None

            return response_event

        finally:
            unsubscribe()

    def get_history(
        self,
        topic: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get event history.

        Args:
            topic: Filter by topic (optional).
            limit: Maximum events to return.

        Returns:
            List of events (newest first).
        """
        events = self._history.copy()

        if topic:
            events = [e for e in events if e.topic == topic]

        return list(reversed(events[-limit:]))

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()


# Global event bus instance
_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _global_bus
    _global_bus = None

