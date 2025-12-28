"""Health checking utilities for AI Glasses Platform."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable

from aiglasses.common.logging import get_logger


class HealthStatus(Enum):
    """Health status enum."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Individual health check definition."""

    name: str
    check_fn: Callable[[], Awaitable[bool]]
    timeout_seconds: float = 5.0
    critical: bool = True  # If False, failure only causes DEGRADED status


@dataclass
class HealthResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AggregatedHealth:
    """Aggregated health status from multiple checks."""

    status: HealthStatus
    checks: list[HealthResult]
    timestamp: float = field(default_factory=time.time)
    message: str = ""


class HealthChecker:
    """Health checker that aggregates multiple health checks."""

    def __init__(self, service_name: str) -> None:
        """Initialize the health checker.

        Args:
            service_name: Name of the service being checked.
        """
        self.service_name = service_name
        self._checks: list[HealthCheck] = []
        self._last_result: AggregatedHealth | None = None
        self.logger = get_logger("health_checker", service=service_name)

    def add_check(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[bool]],
        timeout_seconds: float = 5.0,
        critical: bool = True,
    ) -> None:
        """Add a health check.

        Args:
            name: Check name.
            check_fn: Async function returning True if healthy.
            timeout_seconds: Timeout for the check.
            critical: If True, failure causes UNHEALTHY status.
        """
        self._checks.append(
            HealthCheck(
                name=name,
                check_fn=check_fn,
                timeout_seconds=timeout_seconds,
                critical=critical,
            )
        )

    def remove_check(self, name: str) -> None:
        """Remove a health check by name."""
        self._checks = [c for c in self._checks if c.name != name]

    async def _run_check(self, check: HealthCheck) -> HealthResult:
        """Run a single health check.

        Args:
            check: Health check to run.

        Returns:
            Health check result.
        """
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                check.check_fn(),
                timeout=check.timeout_seconds,
            )
            latency_ms = (time.time() - start_time) * 1000

            if result:
                return HealthResult(
                    name=check.name,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                )
            else:
                return HealthResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY if check.critical else HealthStatus.DEGRADED,
                    message="Check returned False",
                    latency_ms=latency_ms,
                )

        except asyncio.TimeoutError:
            return HealthResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY if check.critical else HealthStatus.DEGRADED,
                message=f"Timeout after {check.timeout_seconds}s",
                latency_ms=check.timeout_seconds * 1000,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY if check.critical else HealthStatus.DEGRADED,
                message=str(e),
                latency_ms=latency_ms,
            )

    async def check(self) -> AggregatedHealth:
        """Run all health checks and aggregate results.

        Returns:
            Aggregated health status.
        """
        if not self._checks:
            return AggregatedHealth(
                status=HealthStatus.HEALTHY,
                checks=[],
                message="No checks configured",
            )

        # Run all checks concurrently
        results = await asyncio.gather(
            *[self._run_check(check) for check in self._checks]
        )

        # Aggregate status
        has_unhealthy = any(r.status == HealthStatus.UNHEALTHY for r in results)
        has_degraded = any(r.status == HealthStatus.DEGRADED for r in results)

        if has_unhealthy:
            status = HealthStatus.UNHEALTHY
            unhealthy_checks = [r.name for r in results if r.status == HealthStatus.UNHEALTHY]
            message = f"Unhealthy checks: {', '.join(unhealthy_checks)}"
        elif has_degraded:
            status = HealthStatus.DEGRADED
            degraded_checks = [r.name for r in results if r.status == HealthStatus.DEGRADED]
            message = f"Degraded checks: {', '.join(degraded_checks)}"
        else:
            status = HealthStatus.HEALTHY
            message = "All checks passed"

        self._last_result = AggregatedHealth(
            status=status,
            checks=list(results),
            message=message,
        )

        return self._last_result

    @property
    def last_result(self) -> AggregatedHealth | None:
        """Get the last health check result."""
        return self._last_result


async def check_grpc_service(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if a gRPC service is reachable.

    Args:
        host: Service host.
        port: Service port.
        timeout: Connection timeout.

    Returns:
        True if service is reachable.
    """
    import grpc
    from grpc import aio as grpc_aio

    try:
        channel = grpc_aio.insecure_channel(
            f"{host}:{port}",
            options=[("grpc.enable_http_proxy", 0)],
        )
        await asyncio.wait_for(
            channel.channel_ready(),
            timeout=timeout,
        )
        await channel.close()
        return True
    except Exception:
        return False


async def check_http_endpoint(url: str, timeout: float = 5.0) -> bool:
    """Check if an HTTP endpoint is reachable.

    Args:
        url: HTTP URL to check.
        timeout: Request timeout.

    Returns:
        True if endpoint returns 2xx status.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return 200 <= response.status_code < 300
    except Exception:
        return False


