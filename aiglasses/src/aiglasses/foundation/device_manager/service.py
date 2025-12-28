"""Device Manager service implementation."""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator

from grpc import aio as grpc_aio

from aiglasses.common import BaseService, HealthStatus, get_logger
from aiglasses.common.events import Event, get_event_bus
from aiglasses.common.health import check_grpc_service
from aiglasses.config import Config


class DeviceManagerService(BaseService):
    """Device Manager service.

    Responsibilities:
    - Boot orchestration
    - Configuration management
    - Health aggregation across all services
    - System metrics collection
    - Event coordination
    """

    def __init__(self, config: Config | None = None, mock_mode: bool = False) -> None:
        super().__init__("device_manager", config, mock_mode)
        self._start_time = time.time()
        self._service_health: dict[str, HealthStatus] = {}
        self._event_bus = get_event_bus()
        self._health_check_task: asyncio.Task | None = None

    @property
    def port(self) -> int:
        return self.config.ports.device_manager

    async def setup(self) -> None:
        """Setup Device Manager resources."""
        self.logger.info("device_manager_setup", mock_mode=self.mock_mode)

        # Start background health monitoring
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())

        # Publish startup event
        await self._event_bus.publish(
            Event(
                topic="system.device_manager.started",
                data={
                    "device_name": self.config.device.name,
                    "mode": self.config.device.mode,
                    "mock_mode": self.mock_mode,
                },
                source="device_manager",
            )
        )

    async def teardown(self) -> None:
        """Teardown Device Manager resources."""
        self.logger.info("device_manager_teardown")

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Publish shutdown event
        await self._event_bus.publish(
            Event(
                topic="system.device_manager.stopped",
                data={},
                source="device_manager",
            )
        )

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services."""
        # Register the DeviceManager gRPC servicer
        from aiglasses.foundation.device_manager.grpc_servicer import (
            DeviceManagerServicer,
            add_servicer,
        )

        servicer = DeviceManagerServicer(self)
        add_servicer(server, servicer)

    async def _health_monitor_loop(self) -> None:
        """Background task to monitor service health."""
        services = [
            ("camera", self.config.ports.camera),
            ("audio", self.config.ports.audio),
            ("speech", self.config.ports.speech),
            ("vision", self.config.ports.vision),
            ("llm_gateway", self.config.ports.llm_gateway),
            ("app_runtime", self.config.ports.app_runtime),
            ("storage", self.config.ports.storage),
            ("ota", self.config.ports.ota),
        ]

        while True:
            try:
                for name, port in services:
                    if self.mock_mode:
                        # In mock mode, assume services are healthy
                        self._service_health[name] = HealthStatus.HEALTHY
                    else:
                        # Check if service is reachable
                        is_healthy = await check_grpc_service("localhost", port, timeout=2.0)
                        self._service_health[name] = (
                            HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                        )

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("health_monitor_error", error=str(e))
                await asyncio.sleep(5)

    async def get_system_health(self) -> dict:
        """Get aggregated system health.

        Returns:
            System health information.
        """
        # Determine overall status
        if not self._service_health:
            overall_status = HealthStatus.UNKNOWN
        elif all(s == HealthStatus.HEALTHY for s in self._service_health.values()):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in self._service_health.values()):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        # Get system resources
        resources = await self._get_system_resources()

        return {
            "overall_status": overall_status.value,
            "services": {name: status.value for name, status in self._service_health.items()},
            "resources": resources,
            "uptime_seconds": int(time.time() - self._start_time),
            "timestamp": time.time(),
        }

    async def _get_system_resources(self) -> dict:
        """Get system resource usage."""
        try:
            # Read CPU usage
            with open("/proc/stat") as f:
                cpu_line = f.readline()
            cpu_values = list(map(int, cpu_line.split()[1:]))
            cpu_total = sum(cpu_values)
            cpu_idle = cpu_values[3]
            cpu_percent = 100 * (1 - cpu_idle / cpu_total) if cpu_total > 0 else 0

            # Read memory usage
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(":")] = int(parts[1])

            mem_total = meminfo.get("MemTotal", 1)
            mem_available = meminfo.get("MemAvailable", mem_total)
            mem_percent = 100 * (1 - mem_available / mem_total)

            # Read disk usage
            statvfs = os.statvfs("/")
            disk_total = statvfs.f_blocks * statvfs.f_frsize
            disk_free = statvfs.f_bfree * statvfs.f_frsize
            disk_percent = 100 * (1 - disk_free / disk_total) if disk_total > 0 else 0

            # Read temperature (Raspberry Pi specific)
            temperature = 0.0
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    temperature = int(f.read().strip()) / 1000
            except FileNotFoundError:
                pass

            return {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(mem_percent, 1),
                "disk_percent": round(disk_percent, 1),
                "temperature_celsius": round(temperature, 1),
            }
        except Exception as e:
            self.logger.warning("resource_read_error", error=str(e))
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0,
                "temperature_celsius": 0.0,
            }

    async def get_service_status(self, service_name: str) -> dict | None:
        """Get status of a specific service.

        Args:
            service_name: Name of the service.

        Returns:
            Service status or None if not found.
        """
        if service_name not in self._service_health:
            return None

        return {
            "name": service_name,
            "health": self._service_health[service_name].value,
            "port": getattr(self.config.ports, service_name, None),
        }

    async def get_config(self) -> dict:
        """Get current system configuration.

        Returns:
            Configuration as dictionary.
        """
        return self.config.model_dump()

    async def subscribe_events(
        self,
        event_types: list[str] | None = None,
    ) -> AsyncIterator[Event]:
        """Subscribe to system events.

        Args:
            event_types: Event types to subscribe to (empty = all).

        Yields:
            System events.
        """
        queue: asyncio.Queue[Event] = asyncio.Queue()

        async def handler(event: Event) -> None:
            await queue.put(event)

        # Subscribe to requested event types or all system events
        pattern = "system.**" if not event_types else "system.*"
        unsubscribe = self._event_bus.subscribe(pattern, handler)

        try:
            while True:
                event = await queue.get()
                if event_types and event.topic not in event_types:
                    continue
                yield event
        finally:
            unsubscribe()


def main() -> None:
    """Entry point for Device Manager service."""
    service = DeviceManagerService()
    service.run()


if __name__ == "__main__":
    main()


