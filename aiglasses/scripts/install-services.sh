#!/bin/bash
# Install AI Glasses systemd services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Installing AI Glasses systemd services..."

# Create aiglasses user if not exists
if ! id -u aiglasses &>/dev/null; then
    echo "Creating aiglasses user..."
    sudo useradd -r -s /bin/false -d /var/lib/aiglasses aiglasses
fi

# Create directories
sudo mkdir -p /etc/aiglasses
sudo mkdir -p /var/lib/aiglasses/apps
sudo mkdir -p /var/log/aiglasses

# Set ownership
sudo chown -R aiglasses:aiglasses /var/lib/aiglasses
sudo chown -R aiglasses:aiglasses /var/log/aiglasses

# Copy config if not exists
if [ ! -f /etc/aiglasses/config.yaml ]; then
    echo "Copying default configuration..."
    sudo cp "$PROJECT_DIR/configs/aiglasses.yaml" /etc/aiglasses/config.yaml
fi

# Copy systemd files
echo "Installing systemd unit files..."
sudo cp "$PROJECT_DIR/systemd/"*.service /etc/systemd/system/
sudo cp "$PROJECT_DIR/systemd/"*.target /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services
echo "Enabling services..."
sudo systemctl enable aiglasses.target
sudo systemctl enable aiglasses-device-manager.service

echo "Installation complete!"
echo ""
echo "To start all services:"
echo "  sudo systemctl start aiglasses.target"
echo ""
echo "To check status:"
echo "  aigctl status"


