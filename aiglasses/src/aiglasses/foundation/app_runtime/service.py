"""App Runtime service implementation."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

import yaml
from grpc import aio as grpc_aio

from aiglasses.common import BaseService, get_logger
from aiglasses.common.events import Event, get_event_bus
from aiglasses.config import Config


class AppState(Enum):
    """App state enum."""

    INSTALLED = "installed"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CRASHED = "crashed"
    ERROR = "error"


class Permission(Enum):
    """Permission enum."""

    CAMERA = "camera"
    MICROPHONE = "microphone"
    BLUETOOTH = "bluetooth"
    NETWORK = "network"
    STORAGE = "storage"
    SYSTEM = "system"


@dataclass
class AppManifest:
    """App manifest."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    entrypoint: str = ""
    required_permissions: list[Permission] = field(default_factory=list)
    optional_permissions: list[Permission] = field(default_factory=list)
    resource_limits: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)


@dataclass
class AppInfo:
    """App information."""

    id: str
    manifest: AppManifest
    state: AppState = AppState.INSTALLED
    granted_permissions: list[Permission] = field(default_factory=list)
    installed_at: float = field(default_factory=time.time)
    last_started: float | None = None
    pid: int | None = None


@dataclass
class LogEntry:
    """Log entry."""

    timestamp: float
    level: str
    message: str
    app_id: str
    fields: dict = field(default_factory=dict)


class AppRuntimeService(BaseService):
    """App Runtime service.

    Responsibilities:
    - App lifecycle management
    - Permission enforcement
    - Resource limits
    """

    def __init__(self, config: Config | None = None, mock_mode: bool = False) -> None:
        super().__init__("app_runtime", config, mock_mode)
        self._apps: dict[str, AppInfo] = {}
        self._app_processes: dict[str, asyncio.subprocess.Process] = {}
        self._event_bus = get_event_bus()
        self._apps_dir = Path("/var/lib/aiglasses/apps")

    @property
    def port(self) -> int:
        return self.config.ports.app_runtime

    async def setup(self) -> None:
        """Setup App Runtime service."""
        self.logger.info("app_runtime_setup", mock_mode=self.mock_mode)

        # Create apps directory
        if not self.mock_mode:
            self._apps_dir.mkdir(parents=True, exist_ok=True)

        # Load installed apps
        await self._load_installed_apps()

    async def teardown(self) -> None:
        """Teardown App Runtime service."""
        self.logger.info("app_runtime_teardown")

        # Stop all running apps
        for app_id in list(self._app_processes.keys()):
            await self.stop_app(app_id, force=True)

    def register_services(self, server: grpc_aio.Server) -> None:
        """Register gRPC services."""
        from aiglasses.foundation.app_runtime.grpc_servicer import (
            AppRuntimeServicer,
            add_servicer,
        )

        servicer = AppRuntimeServicer(self)
        add_servicer(server, servicer)

    async def _load_installed_apps(self) -> None:
        """Load installed apps from disk."""
        if self.mock_mode:
            # Add mock app
            self._apps["what-am-i-seeing"] = AppInfo(
                id="what-am-i-seeing",
                manifest=AppManifest(
                    name="What Am I Seeing",
                    version="0.1.0",
                    description="Visual question answering app",
                    entrypoint="python -m examples.what_am_i_seeing",
                    required_permissions=[
                        Permission.CAMERA,
                        Permission.MICROPHONE,
                        Permission.NETWORK,
                    ],
                ),
                state=AppState.INSTALLED,
                granted_permissions=[
                    Permission.CAMERA,
                    Permission.MICROPHONE,
                    Permission.NETWORK,
                ],
            )
            return

        if not self._apps_dir.exists():
            return

        for app_dir in self._apps_dir.iterdir():
            if app_dir.is_dir():
                manifest_path = app_dir / "manifest.yaml"
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest_data = yaml.safe_load(f)
                        manifest = AppManifest(
                            name=manifest_data.get("name", app_dir.name),
                            version=manifest_data.get("version", "0.0.0"),
                            description=manifest_data.get("description", ""),
                            author=manifest_data.get("author", ""),
                            entrypoint=manifest_data.get("entrypoint", ""),
                            required_permissions=[
                                Permission(p) for p in manifest_data.get("required_permissions", [])
                            ],
                            optional_permissions=[
                                Permission(p) for p in manifest_data.get("optional_permissions", [])
                            ],
                            resource_limits=manifest_data.get("resource_limits", {}),
                            env=manifest_data.get("env", {}),
                        )
                        self._apps[app_dir.name] = AppInfo(
                            id=app_dir.name,
                            manifest=manifest,
                        )
                    except Exception as e:
                        self.logger.warning("failed_to_load_app", app=app_dir.name, error=str(e))

    async def install_app(
        self,
        app_id: str,
        manifest: AppManifest,
        force: bool = False,
    ) -> AppInfo:
        """Install an app.

        Args:
            app_id: App identifier.
            manifest: App manifest.
            force: Overwrite existing app.

        Returns:
            App info.
        """
        if app_id in self._apps and not force:
            raise ValueError(f"App {app_id} already installed")

        app_info = AppInfo(
            id=app_id,
            manifest=manifest,
            state=AppState.INSTALLED,
        )
        self._apps[app_id] = app_info

        # Publish event
        await self._event_bus.publish(
            Event(
                topic="app.installed",
                data={"app_id": app_id, "name": manifest.name},
                source="app_runtime",
            )
        )

        return app_info

    async def uninstall_app(self, app_id: str, remove_data: bool = False) -> None:
        """Uninstall an app.

        Args:
            app_id: App identifier.
            remove_data: Remove app data.
        """
        if app_id not in self._apps:
            raise ValueError(f"App {app_id} not found")

        # Stop if running
        if app_id in self._app_processes:
            await self.stop_app(app_id, force=True)

        del self._apps[app_id]

        await self._event_bus.publish(
            Event(
                topic="app.uninstalled",
                data={"app_id": app_id},
                source="app_runtime",
            )
        )

    async def start_app(
        self,
        app_id: str,
        env_override: dict | None = None,
    ) -> int:
        """Start an app.

        Args:
            app_id: App identifier.
            env_override: Environment variable overrides.

        Returns:
            Process ID.
        """
        if app_id not in self._apps:
            raise ValueError(f"App {app_id} not found")

        app_info = self._apps[app_id]

        if app_id in self._app_processes:
            raise ValueError(f"App {app_id} already running")

        # Check permissions
        missing = set(app_info.manifest.required_permissions) - set(app_info.granted_permissions)
        if missing:
            raise PermissionError(f"Missing permissions: {missing}")

        app_info.state = AppState.STARTING

        # Setup environment
        env = os.environ.copy()
        env.update(app_info.manifest.env)
        if env_override:
            env.update(env_override)
        env["AIGLASSES_APP_ID"] = app_id
        env["AIGLASSES_MOCK_MODE"] = "1" if self.mock_mode else "0"

        # Start process
        try:
            process = await asyncio.create_subprocess_shell(
                app_info.manifest.entrypoint,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._app_processes[app_id] = process
            app_info.pid = process.pid
            app_info.state = AppState.RUNNING
            app_info.last_started = time.time()

            # Monitor process
            asyncio.create_task(self._monitor_process(app_id, process))

            await self._event_bus.publish(
                Event(
                    topic="app.started",
                    data={"app_id": app_id, "pid": process.pid},
                    source="app_runtime",
                )
            )

            return process.pid

        except Exception as e:
            app_info.state = AppState.ERROR
            raise

    async def _monitor_process(
        self,
        app_id: str,
        process: asyncio.subprocess.Process,
    ) -> None:
        """Monitor an app process."""
        await process.wait()

        if app_id in self._apps:
            app_info = self._apps[app_id]
            if process.returncode == 0:
                app_info.state = AppState.STOPPED
            else:
                app_info.state = AppState.CRASHED
                await self._event_bus.publish(
                    Event(
                        topic="app.crashed",
                        data={"app_id": app_id, "exit_code": process.returncode},
                        source="app_runtime",
                    )
                )

        if app_id in self._app_processes:
            del self._app_processes[app_id]

    async def stop_app(
        self,
        app_id: str,
        force: bool = False,
        timeout: int = 5,
    ) -> None:
        """Stop an app.

        Args:
            app_id: App identifier.
            force: Force kill (SIGKILL).
            timeout: Shutdown timeout.
        """
        if app_id not in self._app_processes:
            return

        process = self._app_processes[app_id]
        app_info = self._apps.get(app_id)
        if app_info:
            app_info.state = AppState.STOPPING

        try:
            if force:
                process.kill()
            else:
                process.terminate()

            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

        if app_info:
            app_info.state = AppState.STOPPED
            app_info.pid = None

        if app_id in self._app_processes:
            del self._app_processes[app_id]

        await self._event_bus.publish(
            Event(
                topic="app.stopped",
                data={"app_id": app_id},
                source="app_runtime",
            )
        )

    def get_app(self, app_id: str) -> AppInfo | None:
        """Get app info."""
        return self._apps.get(app_id)

    def list_apps(self) -> list[AppInfo]:
        """List all apps."""
        return list(self._apps.values())

    async def grant_permission(
        self,
        app_id: str,
        permission: Permission,
    ) -> None:
        """Grant a permission to an app."""
        if app_id not in self._apps:
            raise ValueError(f"App {app_id} not found")

        app_info = self._apps[app_id]
        if permission not in app_info.granted_permissions:
            app_info.granted_permissions.append(permission)

    async def revoke_permission(
        self,
        app_id: str,
        permission: Permission,
    ) -> None:
        """Revoke a permission from an app."""
        if app_id not in self._apps:
            raise ValueError(f"App {app_id} not found")

        app_info = self._apps[app_id]
        if permission in app_info.granted_permissions:
            app_info.granted_permissions.remove(permission)

    def get_status(self) -> dict:
        """Get runtime status."""
        running = len(self._app_processes)
        installed = len(self._apps)

        return {
            "running_apps": running,
            "installed_apps": installed,
        }


def main() -> None:
    """Entry point for App Runtime service."""
    service = AppRuntimeService()
    service.run()


if __name__ == "__main__":
    main()


