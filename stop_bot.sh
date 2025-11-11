#!/bin/bash
# Script to stop the SocialRobot voice assistant
# Usage: ./stop_bot.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.bot.pid"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}Bot is not running (no PID file found)${NC}"
    exit 0
fi

# Read PID
BOT_PID=$(cat "$PID_FILE")

# Check if process is running
if ! ps -p "$BOT_PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}Bot is not running (PID $BOT_PID not found)${NC}"
    rm "$PID_FILE"
    exit 0
fi

# Stop the bot
echo -e "${YELLOW}Stopping bot (PID $BOT_PID)...${NC}"
kill "$BOT_PID"

# Wait for it to stop (with timeout)
TIMEOUT=10
ELAPSED=0
while ps -p "$BOT_PID" > /dev/null 2>&1; do
    sleep 0.5
    ELAPSED=$((ELAPSED + 1))
    if [ $ELAPSED -ge $((TIMEOUT * 2)) ]; then
        echo -e "${RED}Bot did not stop gracefully, forcing...${NC}"
        kill -9 "$BOT_PID" 2>/dev/null || true
        sleep 1
        break
    fi
done

# Remove PID file
rm -f "$PID_FILE"

echo -e "${GREEN}✓ Bot stopped successfully${NC}"

