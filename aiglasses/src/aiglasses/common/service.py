"""Base service class for Foundation services."""

from __future__ import annotations

import asyncio
import signal
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import grpc
from grpc import aio as grpc_aio

from aiglasses.common.logging import get_logger, setup_logging
from aiglasses.common.health import HealthChecker, HealthStatus
from aiglasses.config import Config, load_config


class ServiceState(Enum):
    """Service state enum."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BaseService(ABC):
    """Base class for all Foundation services.

    Provides common functionality:
    - gRPC server setup
    - Health checking
    - Graceful shutdown
    - Logging
    - Configuration
    """

    def __init__(
        self,
        name: str,
        config: Config | None = None,
        mock_mode: bool = False,
    ) -> None:
        """Initialize the service.

        Args:
            name: Service name.
            config: Configuration object. Loaded from file if None.
            mock_mode: Run in mock mode for testing.
        """
        self.name = name
        self.config = config or load_config()
        self.mock_mode = mock_mode or self.config.mock_mode

        # Setup logging
        setup_logging(
            level=self.config.device.log_level,
            json_output=self.config.device.mode == "production",
            service_name=name,
        )
        self.logger = get_logger(name, service=name)

        # Service state
        self._state = ServiceState.STOPPED
        self._server: grpc_aio.Server | None = None
        self._health_checker = HealthChecker(name)
        self._shutdown_event = asyncio.Event()

    @property
    def state(self) -> ServiceState:
        """Get current service state."""
        return self._state

    @property
    def port(self) -> int:
        """Get the gRPC port for this service."""
        return getattr(self.config.ports, self.name.replace("-", "_").replace("aiglasses_", ""))

    @abstractmethod
    async def setup(self) -> None:
        """Setup service-specific resources.

        Called before the gRPC server starts.
        """
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """Teardown service-specific resources.

        Called after the gRPC server stops.
        """
        pass

    @abstractmethod
    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services on the server.

        Args:
            server: gRPC server to register services on.
        """
        pass

    async def check_health(self) -> HealthStatus:
        """Check service health.

        Override to add service-specific health checks.

        Returns:
            Health status.
        """
        if self._state == ServiceState.RUNNING:
            return HealthStatus.HEALTHY
        elif self._state == ServiceState.ERROR:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED

    async def start(self) -> None:
        """Start the service."""
        self.logger.info("starting_service", port=self.port)
        self._state = ServiceState.STARTING

        try:
            # Setup service resources
            await self.setup()

            # Create and configure gRPC server
            self._server = grpc_aio.server(
                options=[
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ]
            )

            # Register services
            self.register_services(self._server)

            # Add insecure port (TLS handled by reverse proxy in production)
            listen_addr = f"[::]:{self.port}"
            self._server.add_insecure_port(listen_addr)

            # Start server
            await self._server.start()

            self._state = ServiceState.RUNNING
            self.logger.info("service_started", port=self.port)

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except Exception as e:
            self._state = ServiceState.ERROR
            self.logger.exception("service_start_failed", error=str(e))
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the service gracefully."""
        if self._state in (ServiceState.STOPPING, ServiceState.STOPPED):
            return

        self.logger.info("stopping_service")
        self._state = ServiceState.STOPPING

        try:
            # Stop gRPC server with grace period
            if self._server:
                await self._server.stop(grace=5.0)

            # Teardown service resources
            await self.teardown()

            self._state = ServiceState.STOPPED
            self.logger.info("service_stopped")

        except Exception as e:
            self._state = ServiceState.ERROR
            self.logger.exception("service_stop_failed", error=str(e))

    def shutdown(self) -> None:
        """Signal the service to shutdown."""
        self._shutdown_event.set()

    def run(self) -> None:
        """Run the service (blocking).

        Sets up signal handlers and runs the async event loop.
        """
        # Setup signal handlers
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.shutdown)

        try:
            loop.run_until_complete(self.start())
        finally:
            loop.close()


class MockService(BaseService):
    """Mock service for testing."""

    def __init__(self, name: str, config: Config | None = None) -> None:
        super().__init__(name, config, mock_mode=True)
        self._mock_data: dict[str, Any] = {}

    async def setup(self) -> None:
        """Setup mock resources."""
        self.logger.info("mock_service_setup")

    async def teardown(self) -> None:
        """Teardown mock resources."""
        self.logger.info("mock_service_teardown")

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register mock gRPC services."""
        pass

    def set_mock_data(self, key: str, value: Any) -> None:
        """Set mock data for testing."""
        self._mock_data[key] = value

    def get_mock_data(self, key: str, default: Any = None) -> Any:
        """Get mock data."""
        return self._mock_data.get(key, default)


