"""SDK Events - pub/sub event system."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
import time
import uuid

from aiglasses.common.logging import get_logger


@dataclass
class Event:
    """Event message."""

    topic: str
    data: dict[str, Any]
    source: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Local event bus for pub/sub messaging within an app.

    Example:
        bus = EventBus()

        @bus.subscribe("user.question")
        async def handle_question(event):
            print(f"Question: {event.data['text']}")

        await bus.publish(Event(
            topic="user.question",
            data={"text": "What am I seeing?"},
            source="my-app"
        ))
    """

    def __init__(self) -> None:
        """Initialize event bus."""
        self._handlers: dict[str, list[EventHandler]] = {}
        self._wildcard_handlers: list[tuple[str, EventHandler]] = []
        self.logger = get_logger("sdk.events")

    def subscribe(
        self,
        topic: str,
        handler: EventHandler | None = None,
    ) -> Callable[[EventHandler], EventHandler] | Callable[[], None]:
        """Subscribe to events on a topic.

        Can be used as a decorator or called directly.

        Args:
            topic: Topic to subscribe to. Supports wildcards (* and **).
            handler: Handler function (optional, for decorator use).

        Returns:
            Decorator (when handler is None) or unsubscribe function.

        Example:
            # As decorator
            @bus.subscribe("audio.*")
            async def handle_audio(event):
                print(event)

            # Direct call
            unsub = bus.subscribe("vision.detections", my_handler)
            # Later: unsub()
        """
        if handler is not None:
            self._register(topic, handler)

            def unsubscribe() -> None:
                self._unregister(topic, handler)

            return unsubscribe

        def decorator(fn: EventHandler) -> EventHandler:
            self._register(topic, fn)
            return fn

        return decorator

    def _register(self, topic: str, handler: EventHandler) -> None:
        """Register a handler for a topic."""
        if "*" in topic:
            self._wildcard_handlers.append((topic, handler))
        else:
            if topic not in self._handlers:
                self._handlers[topic] = []
            self._handlers[topic].append(handler)

    def _unregister(self, topic: str, handler: EventHandler) -> None:
        """Unregister a handler from a topic."""
        if "*" in topic:
            self._wildcard_handlers = [
                (t, h) for t, h in self._wildcard_handlers
                if not (t == topic and h == handler)
            ]
        elif topic in self._handlers:
            self._handlers[topic] = [
                h for h in self._handlers[topic] if h != handler
            ]

    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches a wildcard pattern."""
        topic_parts = topic.split(".")
        pattern_parts = pattern.split(".")

        i, j = 0, 0
        while i < len(topic_parts) and j < len(pattern_parts):
            if pattern_parts[j] == "**":
                if j == len(pattern_parts) - 1:
                    return True
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

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish.

        Example:
            await bus.publish(Event(
                topic="vision.detection",
                data={"label": "person", "confidence": 0.9},
                source="my-app"
            ))
        """
        self.logger.debug(
            "publishing_event",
            topic=event.topic,
            event_id=event.event_id,
        )

        # Get exact topic handlers
        handlers = self._handlers.get(event.topic, []).copy()

        # Get wildcard handlers
        for pattern, handler in self._wildcard_handlers:
            if self._matches_pattern(event.topic, pattern):
                handlers.append(handler)

        # Dispatch to all handlers
        if handlers:
            await asyncio.gather(
                *[self._safe_dispatch(h, event) for h in handlers],
                return_exceptions=True,
            )

    async def _safe_dispatch(self, handler: EventHandler, event: Event) -> None:
        """Safely dispatch event to handler."""
        try:
            await handler(event)
        except Exception as e:
            self.logger.exception(
                "event_handler_error",
                topic=event.topic,
                error=str(e),
            )


# Module-level convenience functions
_default_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the default event bus."""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def subscribe(topic: str) -> Callable[[EventHandler], EventHandler]:
    """Subscribe to events on the default bus.

    Example:
        @subscribe("audio.wake")
        async def on_wake(event):
            print("Wake word detected!")
    """
    return get_event_bus().subscribe(topic)


async def publish(event: Event) -> None:
    """Publish event to the default bus."""
    await get_event_bus().publish(event)

