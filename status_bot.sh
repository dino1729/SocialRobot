#!/bin/bash
# Script to check the status of the SocialRobot voice assistant
# Usage: ./status_bot.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.bot.pid"
LOG_FILE="$SCRIPT_DIR/bot.log"

echo -e "${BLUE}SocialRobot Voice Assistant Status${NC}"
echo "=================================="
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "Status: ${RED}Not running${NC}"
    exit 0
fi

# Read PID
BOT_PID=$(cat "$PID_FILE")

# Check if process is running
if ! ps -p "$BOT_PID" > /dev/null 2>&1; then
    echo -e "Status: ${RED}Not running${NC} (stale PID file)"
    echo -e "PID:    ${YELLOW}$BOT_PID (not found)${NC}"
    exit 0
fi

# Get process info
PROCESS_INFO=$(ps -p "$BOT_PID" -o pid,ppid,etime,cmd --no-headers 2>/dev/null)

echo -e "Status:  ${GREEN}Running ✓${NC}"
echo -e "PID:     ${GREEN}$BOT_PID${NC}"

# Parse and display process info
if [ -n "$PROCESS_INFO" ]; then
    UPTIME=$(echo "$PROCESS_INFO" | awk '{print $3}')
    COMMAND=$(echo "$PROCESS_INFO" | awk '{$1=$2=$3=""; print $0}' | sed 's/^[ \t]*//')
    echo -e "Uptime:  ${GREEN}$UPTIME${NC}"
    echo -e "Command: ${GREEN}$COMMAND${NC}"
fi

# Show log file info
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    echo -e "Log:     ${GREEN}$LOG_FILE${NC} (${LOG_SIZE})"
    
    echo ""
    echo -e "${BLUE}Recent log entries (last 10 lines):${NC}"
    echo "-----------------------------------"
    tail -n 10 "$LOG_FILE"
else
    echo -e "Log:     ${YELLOW}Not found${NC}"
fi

echo ""
echo "To stop the bot: ./stop_bot.sh"
echo "To view logs:    tail -f $LOG_FILE"

