# AI Glasses Platform

Open-source AI glasses platform built on Raspberry Pi 5 with IMX500 AI Camera and optional Hailo AI HAT+.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User                                        │
│                    (AirPods + Glasses-mounted Camera)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │      Raspberry Pi 5           │
                    │      (Pocket Hub)             │
                    └───────────────┬───────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                         Foundation Layer                                 │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┤
│ Device Mgr  │ Camera Svc  │ Audio Svc   │ Speech Svc  │ Vision Svc      │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────────┤
│ LLM Gateway │ App Runtime │ Storage     │ OTA         │ Observability   │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                              SDK Layer                                   │
│         audio / vision / llm / app lifecycle / events                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                           Applications                                   │
│                    (e.g., "What am I seeing?")                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

- Raspberry Pi 5 (4GB+ RAM recommended)
- Raspberry Pi AI Camera (Sony IMX500)
- Raspberry Pi AI HAT+ (Hailo) - optional for advanced CV
- AirPods or compatible Bluetooth headset
- Pi OS Bookworm 64-bit

## Quick Start

### Prerequisites

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv libcamera-dev \
    bluez pipewire wireplumber python3-dbus python3-gi

# Clone the repository
git clone https://github.com/michaltakac/juicy-ai-glasses.git
cd aiglasses

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Run with Mock Services (Development)

```bash
# Start foundation services in mock mode
aigctl start --mock

# Run the example app
python -m examples.what_am_i_seeing --mock
```

### Run on Real Hardware

```bash
# Configure the system
sudo cp configs/aiglasses.yaml /etc/aiglasses/config.yaml
# Edit config as needed

# Install systemd services
sudo ./scripts/install-services.sh

# Start all services
sudo systemctl start aiglasses.target

# Run the example app
aigctl app start what-am-i-seeing
```

## Project Structure

```
aiglasses/
├── foundation/           # Core platform services
│   ├── device_manager/   # Boot orchestration, health
│   ├── camera/           # Frame capture, IMX500 inference
│   ├── audio/            # Bluetooth audio, mic/speaker
│   ├── speech/           # STT, TTS, wake word
│   ├── vision/           # Object detection pipeline
│   ├── llm_gateway/      # Remote LLM broker
│   ├── app_runtime/      # App sandbox and lifecycle
│   ├── storage/          # Privacy-aware storage
│   └── ota/              # Update mechanism
├── sdk/
│   └── python/           # Python SDK
├── examples/
│   └── what_am_i_seeing/ # Example app
├── proto/                # gRPC protocol definitions
├── configs/              # Configuration templates
├── systemd/              # Systemd service files
├── tests/                # Test suite
├── tools/                # Development utilities
└── docs/                 # Documentation
```

## Configuration

Main config file: `/etc/aiglasses/config.yaml`

```yaml
device:
  name: "my-aiglasses"
  mode: "production"  # or "development"

audio:
  bluetooth_device: "AirPods"
  fallback_enabled: true

camera:
  resolution: [1920, 1080]
  fps: 30
  imx500_model: "mobilenet_v2"

llm:
  provider: "lan"  # "lan", "openai", "anthropic"
  lan_endpoint: "http://macbook.local:8080/v1"
  # api_key: from environment AIGLASSES_LLM_API_KEY

privacy:
  retention_seconds: 60
  encrypt_storage: false
  capture_indicator: true
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit

# Integration tests (mock mode)
pytest tests/integration --mock

# Hardware-in-the-loop tests (requires real Pi + hardware)
pytest tests/integration --hil
```

### Building

```bash
# Build Python package
python -m build

# Build Debian package
./scripts/build-deb.sh
```

## License

MIT License - see LICENSE file for details.

## Contributing

See CONTRIBUTING.md for guidelines.


