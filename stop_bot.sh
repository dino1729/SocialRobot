#!/bin/bash
# Script to stop the SocialRobot voice assistant
# Usage: ./stop_bot.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.bot.pid"

stop_by_pid() {
    local pid=$1
    echo -e "${YELLOW}Stopping bot (PID $pid)...${NC}"
    kill "$pid" 2>/dev/null || true

    # Wait for it to stop (with timeout)
    local timeout=10
    local elapsed=0
    while ps -p "$pid" > /dev/null 2>&1; do
        sleep 0.5
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $((timeout * 2)) ]; then
            echo -e "${RED}Bot did not stop gracefully, forcing...${NC}"
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
            break
        fi
    done
}

# Try to stop via PID file first
if [ -f "$PID_FILE" ]; then
    BOT_PID=$(cat "$PID_FILE")
    if ps -p "$BOT_PID" > /dev/null 2>&1; then
        stop_by_pid "$BOT_PID"
        rm -f "$PID_FILE"
        echo -e "${GREEN}✓ Bot stopped successfully${NC}"
        exit 0
    else
        echo -e "${YELLOW}PID $BOT_PID not running, removing stale PID file${NC}"
        rm -f "$PID_FILE"
    fi
fi

# Fallback: find and kill any running main.py processes in this directory
MAIN_PIDS=$(pgrep -f "python.*main\.py" 2>/dev/null || true)
if [ -n "$MAIN_PIDS" ]; then
    echo -e "${YELLOW}Found orphaned bot process(es), stopping...${NC}"
    for pid in $MAIN_PIDS; do
        stop_by_pid "$pid"
    done
    echo -e "${GREEN}✓ Bot stopped successfully${NC}"
    exit 0
fi

echo -e "${YELLOW}Bot is not running${NC}"

