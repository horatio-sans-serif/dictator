#!/bin/bash
# Install dictator web server as a launchd service (auto-start on login)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_SRC="$PROJECT_DIR/etc/com.dictator.server.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.dictator.server.plist"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

UV_PATH=$(which uv)

# Check for MISTRAL_API_KEY
if [ -z "$MISTRAL_API_KEY" ]; then
    echo "Error: MISTRAL_API_KEY environment variable is required"
    echo "Get one at https://console.mistral.ai/"
    exit 1
fi

# Create LaunchAgents directory if needed
mkdir -p "$HOME/Library/LaunchAgents"

# Stop existing service if running
if launchctl list | grep -q com.dictator.server; then
    echo "Stopping existing dictator server..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Copy and customize plist
echo "Installing launchd plist..."
sed -e "s|__UV_PATH__|$UV_PATH|g" \
    -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__MISTRAL_API_KEY__|$MISTRAL_API_KEY|g" \
    "$PLIST_SRC" > "$PLIST_DST"

# Load the service
echo "Starting dictator server..."
launchctl load "$PLIST_DST"

echo "Done! Dictator web UI will start automatically on login."
echo "  URL: http://localhost:8377"
echo "  Check status: launchctl list | grep dictator"
echo "  View logs: tail -f /tmp/dictator-server.log"
