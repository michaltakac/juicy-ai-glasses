"""SDK App Manifest - app metadata and permissions."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Permission(str, Enum):
    """App permissions."""

    CAMERA = "camera"
    MICROPHONE = "microphone"
    BLUETOOTH = "bluetooth"
    NETWORK = "network"
    STORAGE = "storage"
    SYSTEM = "system"


class ResourceLimits(BaseModel):
    """Resource limits for apps."""

    memory_mb: int = 256
    cpu_percent: float = 50.0
    storage_mb: int = 100
    network_rate_kbps: int = 0  # 0 = unlimited


class AppManifest(BaseModel):
    """App manifest definition.

    The manifest file (manifest.yaml) defines metadata, permissions,
    and configuration for an AI Glasses app.

    Example manifest.yaml:

        name: What Am I Seeing
        version: 1.0.0
        description: Visual question answering app
        author: AI Glasses Team

        entrypoint: python -m what_am_i_seeing

        permissions:
          required:
            - camera
            - microphone
            - network
          optional:
            - storage

        resources:
          memory_mb: 256
          cpu_percent: 50

        env:
          LOG_LEVEL: INFO
    """

    name: str
    version: str
    description: str = ""
    author: str = ""
    entrypoint: str = ""

    required_permissions: list[Permission] = Field(default_factory=list)
    optional_permissions: list[Permission] = Field(default_factory=list)

    resources: ResourceLimits = Field(default_factory=ResourceLimits)

    env: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)

    # Runtime-assigned fields
    app_id: str = ""
    install_path: str = ""

    @classmethod
    def from_yaml(cls, path: Path | str) -> AppManifest:
        """Load manifest from YAML file.

        Args:
            path: Path to manifest.yaml file.

        Returns:
            Loaded manifest.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Handle nested permissions structure
        permissions = data.get("permissions", {})
        if isinstance(permissions, dict):
            data["required_permissions"] = [
                Permission(p) for p in permissions.get("required", [])
            ]
            data["optional_permissions"] = [
                Permission(p) for p in permissions.get("optional", [])
            ]
            del data["permissions"]

        # Handle resources
        resources = data.get("resources", {})
        if resources:
            data["resources"] = ResourceLimits(**resources)

        return cls(**data)

    def to_yaml(self, path: Path | str) -> None:
        """Save manifest to YAML file.

        Args:
            path: Path to save manifest.yaml.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "entrypoint": self.entrypoint,
            "permissions": {
                "required": [p.value for p in self.required_permissions],
                "optional": [p.value for p in self.optional_permissions],
            },
            "resources": {
                "memory_mb": self.resources.memory_mb,
                "cpu_percent": self.resources.cpu_percent,
                "storage_mb": self.resources.storage_mb,
            },
            "env": self.env,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def has_permission(self, permission: Permission) -> bool:
        """Check if app requires or optionally uses a permission."""
        return (
            permission in self.required_permissions
            or permission in self.optional_permissions
        )

    def validate_permissions(self, granted: list[Permission]) -> list[Permission]:
        """Validate granted permissions against requirements.

        Args:
            granted: List of granted permissions.

        Returns:
            List of missing required permissions.
        """
        granted_set = set(granted)
        required_set = set(self.required_permissions)
        return list(required_set - granted_set)


def create_manifest(
    name: str,
    version: str = "0.1.0",
    description: str = "",
    author: str = "",
    entrypoint: str = "",
    permissions: list[str] | None = None,
) -> AppManifest:
    """Create a new app manifest.

    Args:
        name: App name.
        version: Version string.
        description: App description.
        author: Author name.
        entrypoint: Command to run the app.
        permissions: Required permission names.

    Returns:
        New manifest.

    Example:
        manifest = create_manifest(
            name="My App",
            version="1.0.0",
            entrypoint="python main.py",
            permissions=["camera", "microphone"]
        )
    """
    required_permissions = []
    if permissions:
        required_permissions = [Permission(p) for p in permissions]

    return AppManifest(
        name=name,
        version=version,
        description=description,
        author=author,
        entrypoint=entrypoint,
        required_permissions=required_permissions,
    )


