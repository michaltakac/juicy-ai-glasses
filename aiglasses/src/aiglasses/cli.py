"""AI Glasses CLI - aigctl command line tool."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from aiglasses import __version__
from aiglasses.config import load_config, Config
from aiglasses.common.grpc_utils import GrpcClient

app = typer.Typer(
    name="aigctl",
    help="AI Glasses Control CLI",
    no_args_is_help=True,
)
console = Console()


def get_config() -> Config:
    """Get configuration."""
    return load_config()


async def call_service(service: str, method: str, request: dict = None) -> dict:
    """Call a Foundation service.

    Args:
        service: Service name (e.g., "device_manager").
        method: Method name.
        request: Request data.

    Returns:
        Response data.
    """
    config = get_config()
    port = getattr(config.ports, service, None)
    if port is None:
        raise ValueError(f"Unknown service: {service}")

    service_map = {
        "device_manager": "aiglasses.device_manager.DeviceManagerService",
        "camera": "aiglasses.camera.CameraService",
        "audio": "aiglasses.audio.AudioService",
        "speech": "aiglasses.speech.SpeechService",
        "vision": "aiglasses.vision.VisionService",
        "llm_gateway": "aiglasses.llm_gateway.LLMGatewayService",
        "app_runtime": "aiglasses.app_runtime.AppRuntimeService",
        "storage": "aiglasses.storage.StorageService",
        "ota": "aiglasses.ota.OTAService",
    }

    service_name = service_map.get(service)
    if not service_name:
        raise ValueError(f"Unknown service: {service}")

    async with GrpcClient("localhost", port) as client:
        return await client.call(service_name, method, request or {})


# Status commands
@app.command()
def status():
    """Show system status."""

    async def _status():
        try:
            health = await call_service("device_manager", "GetHealth")

            # Display overall status
            overall = health.get("overall_status", "unknown")
            status_color = {
                "healthy": "green",
                "degraded": "yellow",
                "unhealthy": "red",
            }.get(overall, "white")

            console.print(
                Panel(
                    f"[bold {status_color}]{overall.upper()}[/]",
                    title="System Status",
                )
            )

            # Display services table
            table = Table(title="Services")
            table.add_column("Service", style="cyan")
            table.add_column("Status")
            table.add_column("Health")

            services = health.get("services", {})
            for name, status in services.items():
                status_style = {
                    "healthy": "green",
                    "degraded": "yellow",
                    "unhealthy": "red",
                }.get(status, "white")
                table.add_row(name, "running", f"[{status_style}]{status}[/]")

            console.print(table)

            # Display resources
            resources = health.get("resources", {})
            if resources:
                console.print("\n[bold]Resources:[/]")
                console.print(f"  CPU: {resources.get('cpu_percent', 0):.1f}%")
                console.print(f"  Memory: {resources.get('memory_percent', 0):.1f}%")
                console.print(f"  Disk: {resources.get('disk_percent', 0):.1f}%")
                console.print(f"  Temperature: {resources.get('temperature_celsius', 0):.1f}°C")

        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            console.print("[dim]Is the device manager service running?[/]")
            sys.exit(1)

    asyncio.run(_status())


@app.command()
def version():
    """Show version information."""

    async def _version():
        try:
            versions = await call_service("ota", "GetVersions")

            console.print(f"[bold]AI Glasses Platform[/] v{__version__}")
            console.print(f"  Foundation: {versions.get('foundation_version', __version__)}")
            console.print(f"  Channel: {versions.get('channel', 'stable')}")

            service_versions = versions.get("service_versions", {})
            if service_versions:
                console.print("\n[bold]Services:[/]")
                for name, ver in service_versions.items():
                    console.print(f"  {name}: {ver}")

        except Exception:
            # Fallback to local version
            console.print(f"[bold]AI Glasses Platform[/] v{__version__}")

    asyncio.run(_version())


# App commands
app_cmd = typer.Typer(help="Manage apps")
app.add_typer(app_cmd, name="app")


@app_cmd.command("list")
def app_list():
    """List installed apps."""

    async def _list():
        try:
            result = await call_service("app_runtime", "ListApps")

            table = Table(title="Installed Apps")
            table.add_column("ID", style="cyan")
            table.add_column("Name")
            table.add_column("Version")
            table.add_column("State")

            for app_info in result.get("apps", []):
                state = app_info.get("state", "unknown")
                state_style = {
                    "running": "green",
                    "stopped": "dim",
                    "crashed": "red",
                }.get(state, "white")

                table.add_row(
                    app_info.get("id", ""),
                    app_info.get("name", ""),
                    app_info.get("version", ""),
                    f"[{state_style}]{state}[/]",
                )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)

    asyncio.run(_list())


@app_cmd.command("start")
def app_start(app_id: str):
    """Start an app."""

    async def _start():
        try:
            result = await call_service("app_runtime", "Start", {"app_id": app_id})

            if result.get("success"):
                console.print(f"[green]Started[/] {app_id} (PID: {result.get('pid')})")
            else:
                console.print(f"[red]Failed:[/] {result.get('message')}")
                sys.exit(1)

        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)

    asyncio.run(_start())


@app_cmd.command("stop")
def app_stop(app_id: str, force: bool = False):
    """Stop an app."""

    async def _stop():
        try:
            result = await call_service(
                "app_runtime", "Stop", {"app_id": app_id, "force": force}
            )

            if result.get("success"):
                console.print(f"[green]Stopped[/] {app_id}")
            else:
                console.print(f"[red]Failed:[/] {result.get('message')}")
                sys.exit(1)

        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)

    asyncio.run(_stop())


# Audio commands
audio_cmd = typer.Typer(help="Audio controls")
app.add_typer(audio_cmd, name="audio")


@audio_cmd.command("status")
def audio_status():
    """Show audio status."""

    async def _status():
        try:
            status = await call_service("audio", "GetStatus")

            device = status.get("current_device", {})
            console.print("[bold]Audio Status[/]")
            console.print(f"  Available: {status.get('available', False)}")
            console.print(f"  Device: {device.get('name', 'None')}")
            console.print(f"  Profile: {status.get('active_profile', 'none')}")
            console.print(f"  Volume: {status.get('volume', 0)}%")
            console.print(f"  Mic Active: {status.get('mic_active', False)}")
            console.print(f"  Playback: {status.get('playback_active', False)}")

        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)

    asyncio.run(_status())


@audio_cmd.command("devices")
def audio_devices():
    """List audio devices."""

    async def _devices():
        try:
            result = await call_service("audio", "ListDevices")

            table = Table(title="Audio Devices")
            table.add_column("Address", style="cyan")
            table.add_column("Name")
            table.add_column("Type")
            table.add_column("Connected")

            for device in result.get("devices", []):
                connected = "✓" if device.get("connected") else ""
                table.add_row(
                    device.get("address", ""),
                    device.get("name", ""),
                    device.get("type", ""),
                    connected,
                )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)

    asyncio.run(_devices())


# Config command
@app.command()
def config(show: bool = True, json_output: bool = False):
    """Show configuration."""
    cfg = get_config()

    if json_output:
        print(json.dumps(cfg.model_dump(), indent=2, default=str))
    else:
        console.print("[bold]Configuration[/]")
        console.print(f"  Device: {cfg.device.name}")
        console.print(f"  Mode: {cfg.device.mode}")
        console.print(f"  Mock Mode: {cfg.mock_mode}")
        console.print(f"\n[bold]LLM[/]")
        console.print(f"  Provider: {cfg.llm.provider}")
        console.print(f"  Model: {cfg.llm.default_model}")
        console.print(f"\n[bold]Privacy[/]")
        console.print(f"  Retention: {cfg.privacy.retention_seconds}s")
        console.print(f"  Encryption: {cfg.privacy.encrypt_storage}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()


