"""Configuration management for AI Glasses Platform."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceConfig(BaseModel):
    """Device configuration."""

    name: str = "aiglasses"
    mode: Literal["production", "development"] = "development"
    log_level: str = "INFO"


class AudioConfig(BaseModel):
    """Audio service configuration."""

    bluetooth_device: str | None = None
    fallback_enabled: bool = True
    sample_rate: int = 16000
    channels: int = 1
    chunk_size_ms: int = 100


class CameraConfig(BaseModel):
    """Camera service configuration."""

    resolution: list[int] = [1920, 1080]
    fps: int = 30
    format: str = "jpeg"
    quality: int = 85
    imx500_model: str = "mobilenet_v2"
    imx500_enabled: bool = True


class SpeechConfig(BaseModel):
    """Speech service configuration."""

    stt_model: str = "tiny"
    stt_language: str = "en"
    tts_voice: str = "default"
    tts_speed: float = 1.0
    wake_words: list[str] = Field(default_factory=lambda: ["hey glasses", "ok glasses"])
    wake_sensitivity: float = 0.5
    vad_enabled: bool = True


class VisionConfig(BaseModel):
    """Vision service configuration."""

    default_model: str = "mobilenet_v2"
    default_confidence: float = 0.5
    hailo_enabled: bool = False
    fusion_enabled: bool = False
    max_concurrent: int = 4


class LLMConfig(BaseModel):
    """LLM Gateway configuration."""

    provider: Literal["lan", "openai", "anthropic", "ollama"] = "lan"
    lan_endpoint: str = "http://localhost:8080/v1"
    openai_endpoint: str = "https://api.openai.com/v1"
    anthropic_endpoint: str = "https://api.anthropic.com/v1"
    ollama_endpoint: str = "http://localhost:11434"
    api_key: str | None = None
    default_model: str = "gpt-4o-mini"
    default_temperature: float = 0.7
    default_max_tokens: int = 1024
    timeout_seconds: int = 30
    retry_count: int = 3
    fallback_provider: str | None = None


class PrivacyConfig(BaseModel):
    """Privacy and storage configuration."""

    retention_seconds: int = 60
    audio_retention_seconds: int = 30
    video_retention_seconds: int = 30
    frame_retention_seconds: int = 60
    encrypt_storage: bool = False
    secure_delete: bool = True
    capture_indicator: bool = True
    max_storage_mb: int = 500


class OTAConfig(BaseModel):
    """OTA update configuration."""

    channel: Literal["stable", "beta", "dev"] = "stable"
    auto_check: bool = True
    check_interval_hours: int = 24
    auto_download: bool = False
    auto_apply: bool = False
    update_server: str = "https://updates.aiglasses.io"


class ServicePorts(BaseModel):
    """Service port configuration."""

    device_manager: int = 50051
    camera: int = 50052
    audio: int = 50053
    speech: int = 50054
    vision: int = 50055
    llm_gateway: int = 50056
    app_runtime: int = 50057
    storage: int = 50058
    ota: int = 50059
    http_api: int = 8080


class Config(BaseSettings):
    """Main configuration for AI Glasses Platform."""

    model_config = SettingsConfigDict(
        env_prefix="AIGLASSES_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    device: DeviceConfig = Field(default_factory=DeviceConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    ota: OTAConfig = Field(default_factory=OTAConfig)
    ports: ServicePorts = Field(default_factory=ServicePorts)

    # Mock mode for development
    mock_mode: bool = False

    @classmethod
    def from_yaml(cls, path: Path | str) -> Config:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def load_config(
    config_path: Path | str | None = None,
    env_override: bool = True,
) -> Config:
    """Load configuration from file and environment.

    Args:
        config_path: Path to config file. If None, searches standard locations.
        env_override: Whether to allow environment variables to override config.

    Returns:
        Loaded configuration.
    """
    # Standard config locations
    search_paths = [
        Path("/etc/aiglasses/config.yaml"),
        Path.home() / ".config" / "aiglasses" / "config.yaml",
        Path("config.yaml"),
        Path("configs/aiglasses.yaml"),
    ]

    if config_path:
        search_paths.insert(0, Path(config_path))

    # Find first existing config file
    config_file: Path | None = None
    for path in search_paths:
        if path.exists():
            config_file = path
            break

    # Load from file or create default
    if config_file:
        config = Config.from_yaml(config_file)
    else:
        config = Config()

    # Apply environment overrides
    if env_override:
        # LLM API key from environment
        api_key = os.environ.get("AIGLASSES_LLM_API_KEY")
        if api_key:
            config.llm.api_key = api_key

        # Mock mode
        if os.environ.get("AIGLASSES_MOCK_MODE", "").lower() in ("1", "true", "yes"):
            config.mock_mode = True

    return config


def get_default_config() -> dict[str, Any]:
    """Get default configuration as dictionary."""
    return Config().model_dump()

