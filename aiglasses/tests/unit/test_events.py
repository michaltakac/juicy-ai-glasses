"""Tests for event bus module."""

import pytest

from aiglasses.common.events import Event, EventBus, get_event_bus, reset_event_bus


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh event bus for each test."""
    return EventBus()


class TestEvent:
    """Tests for Event class."""

    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            topic="test.topic",
            data={"key": "value"},
            source="test",
        )

        assert event.topic == "test.topic"
        assert event.data == {"key": "value"}
        assert event.source == "test"
        assert event.event_id  # Should be auto-generated
        assert event.timestamp > 0

    def test_event_with_correlation_id(self):
        """Test event with correlation ID."""
        event = Event(
            topic="test.topic",
            data={},
            source="test",
            correlation_id="corr-123",
        )

        assert event.correlation_id == "corr-123"


class TestEventBus:
    """Tests for EventBus class."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus: EventBus):
        """Test basic subscribe and publish."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        event_bus.subscribe("test.topic", handler)

        event = Event(topic="test.topic", data={"test": True}, source="test")
        await event_bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0].topic == "test.topic"
        assert received_events[0].data == {"test": True}

    @pytest.mark.asyncio
    async def test_wildcard_subscribe_single(self, event_bus: EventBus):
        """Test single-level wildcard subscription."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        event_bus.subscribe("test.*", handler)

        await event_bus.publish(Event(topic="test.one", data={}, source="test"))
        await event_bus.publish(Event(topic="test.two", data={}, source="test"))
        await event_bus.publish(Event(topic="other.one", data={}, source="test"))

        assert len(received_events) == 2
        assert received_events[0].topic == "test.one"
        assert received_events[1].topic == "test.two"

    @pytest.mark.asyncio
    async def test_wildcard_subscribe_multi(self, event_bus: EventBus):
        """Test multi-level wildcard subscription."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        event_bus.subscribe("test.**", handler)

        await event_bus.publish(Event(topic="test.one", data={}, source="test"))
        await event_bus.publish(Event(topic="test.one.two", data={}, source="test"))
        await event_bus.publish(Event(topic="test.one.two.three", data={}, source="test"))

        assert len(received_events) == 3

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus: EventBus):
        """Test unsubscribe function."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        unsubscribe = event_bus.subscribe("test.topic", handler)

        await event_bus.publish(Event(topic="test.topic", data={}, source="test"))
        assert len(received_events) == 1

        unsubscribe()

        await event_bus.publish(Event(topic="test.topic", data={}, source="test"))
        assert len(received_events) == 1  # Should not receive new events

    @pytest.mark.asyncio
    async def test_handler_error_isolation(self, event_bus: EventBus):
        """Test that handler errors don't affect other handlers."""
        results = []

        async def failing_handler(event: Event):
            raise ValueError("Test error")

        async def working_handler(event: Event):
            results.append(event)

        event_bus.subscribe("test.topic", failing_handler)
        event_bus.subscribe("test.topic", working_handler)

        # Should not raise, working handler should still be called
        await event_bus.publish(Event(topic="test.topic", data={}, source="test"))

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_decorator_subscribe(self, event_bus: EventBus):
        """Test decorator-style subscription."""
        received_events = []

        # Use decorator syntax by calling subscribe with just topic
        # then applying the decorator to the handler
        async def handler(event: Event):
            received_events.append(event)

        # Apply decorator
        event_bus.subscribe("test.topic")(handler)

        await event_bus.publish(Event(topic="test.topic", data={}, source="test"))

        assert len(received_events) == 1


class TestGlobalEventBus:
    """Tests for global event bus."""

    def test_get_event_bus_singleton(self):
        """Test that get_event_bus returns same instance."""
        reset_event_bus()
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_event_bus(self):
        """Test resetting global event bus."""
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()
        assert bus1 is not bus2

