#!/bin/bash
# Uninstall dictator server launchd service

PLIST_DST="$HOME/Library/LaunchAgents/com.dictator.server.plist"

if [ -f "$PLIST_DST" ]; then
    echo "Stopping dictator server..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm "$PLIST_DST"
    echo "Done! Dictator launchd service removed."
else
    echo "Dictator launchd service not installed."
fi
