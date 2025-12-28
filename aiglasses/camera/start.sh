#!/bin/bash
# Start AI Glasses Camera Viewer
# This script runs both the Python backend and Vite frontend

set -e
cd "$(dirname "$0")"

echo "🕶️  AI Glasses Camera Viewer"
echo "============================"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
fi

# Activate Python venv
source ../.venv/bin/activate

# Check aiohttp
if ! python -c "import aiohttp" 2>/dev/null; then
    echo "📦 Installing aiohttp..."
    pip install aiohttp
fi

# Start backend server in background
echo "🎥 Starting camera server on :8765..."
python server.py &
BACKEND_PID=$!

# Give backend time to start
sleep 2

# Start Vite frontend
echo "🌐 Starting Vite dev server on :5173..."
echo ""
echo "Open in browser:"
echo "  Local:   http://localhost:5173"
echo "  Network: http://rpi5kavecany.local:5173"
echo ""

npm run dev &
FRONTEND_PID=$!

# Handle shutdown
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for either to exit
wait


