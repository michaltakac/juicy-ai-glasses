"""Tests for configuration module."""

import os
import tempfile
from pathlib import Path

import pytest

from aiglasses.config import Config, load_config, DeviceConfig, LLMConfig


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = Config()

        assert config.device.name == "aiglasses"
        assert config.device.mode == "development"
        assert config.mock_mode is False
        assert config.llm.provider == "lan"

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config = Config(
            device=DeviceConfig(name="test-device", mode="production"),
            llm=LLMConfig(provider="openai"),
            mock_mode=True,
        )

        assert config.device.name == "test-device"
        assert config.device.mode == "production"
        assert config.llm.provider == "openai"
        assert config.mock_mode is True

    def test_config_to_yaml(self):
        """Test saving config to YAML."""
        config = Config(
            device=DeviceConfig(name="yaml-test"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)

            assert path.exists()

            # Load it back
            loaded = Config.from_yaml(path)
            assert loaded.device.name == "yaml-test"

    def test_config_from_yaml(self):
        """Test loading config from YAML."""
        yaml_content = """
device:
  name: yaml-device
  mode: production
llm:
  provider: openai
  default_model: gpt-4
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_content)

            config = Config.from_yaml(path)

            assert config.device.name == "yaml-device"
            assert config.device.mode == "production"
            assert config.llm.provider == "openai"
            assert config.llm.default_model == "gpt-4"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default(self, tmp_path, monkeypatch):
        """Test loading default config when no file exists."""
        # Change to temp directory so no config files are found
        monkeypatch.chdir(tmp_path)
        config = load_config(config_path="/nonexistent/path.yaml")
        assert config is not None
        assert config.device.name == "aiglasses"

    def test_load_with_env_override(self):
        """Test environment variable overrides."""
        os.environ["AIGLASSES_MOCK_MODE"] = "1"
        os.environ["AIGLASSES_LLM_API_KEY"] = "test-key"

        try:
            config = load_config()
            assert config.mock_mode is True
            assert config.llm.api_key == "test-key"
        finally:
            del os.environ["AIGLASSES_MOCK_MODE"]
            del os.environ["AIGLASSES_LLM_API_KEY"]

    def test_load_from_file(self):
        """Test loading config from file."""
        yaml_content = """
device:
  name: file-config
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_content)

            config = load_config(config_path=path)
            assert config.device.name == "file-config"

