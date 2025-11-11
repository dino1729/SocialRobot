#!/bin/bash
# Script to start the SocialRobot voice assistant in the background
# Usage: ./start_bot.sh [VARIANT] [ADDITIONAL_ARGS...]
#
# VARIANT options:
#   wakeword                      - Basic voice assistant with wake word (Ollama)
#   wakeword_online               - Basic voice assistant with wake word (LiteLLM/OpenAI)
#   wakeword_internetconnected    - Voice assistant with internet tools (Ollama)
#   wakeword_internetconnected_online - Voice assistant with internet tools (LiteLLM/OpenAI) [DEFAULT]
#
# Examples:
#   ./start_bot.sh                                    # Start default (internetconnected_online)
#   ./start_bot.sh wakeword                           # Start basic wakeword version
#   ./start_bot.sh wakeword_online --tts-engine piper # Start online version with Piper TTS
#   ./start_bot.sh wakeword_internetconnected --compute-type int8 --wakeword-threshold 0.6

set -e  # Exit on error

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

# Default variant
VARIANT="${1:-wakeword_internetconnected_online}"

# If first argument is provided and looks like a variant, shift it
if [[ "$1" == "wakeword"* ]]; then
    shift
fi

# Map variant to script name
case "$VARIANT" in
    wakeword)
        SCRIPT="main_wakeword.py"
        ;;
    wakeword_online)
        SCRIPT="main_wakeword_online.py"
        ;;
    wakeword_internetconnected)
        SCRIPT="main_wakeword_internetconnected.py"
        ;;
    wakeword_internetconnected_online)
        SCRIPT="main_wakeword_internetconnected_online.py"
        ;;
    *)
        echo -e "${RED}Error: Unknown variant '$VARIANT'${NC}"
        echo ""
        echo "Available variants:"
        echo "  wakeword                      - Basic voice assistant with wake word (Ollama)"
        echo "  wakeword_online               - Basic voice assistant with wake word (LiteLLM/OpenAI)"
        echo "  wakeword_internetconnected    - Voice assistant with internet tools (Ollama)"
        echo "  wakeword_internetconnected_online - Voice assistant with internet tools (LiteLLM/OpenAI)"
        echo ""
        echo "Usage: ./start_bot.sh [VARIANT] [ADDITIONAL_ARGS...]"
        exit 1
        ;;
esac

# Check if bot is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Bot is already running with PID $OLD_PID${NC}"
        echo "Use ./stop_bot.sh to stop it first, or check the logs:"
        echo "  tail -f $LOG_FILE"
        exit 1
    else
        echo -e "${YELLOW}Removing stale PID file${NC}"
        rm "$PID_FILE"
    fi
fi

# Check if script exists
if [ ! -f "$SCRIPT_DIR/$SCRIPT" ]; then
    echo -e "${RED}Error: Script '$SCRIPT' not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Make sure environment variables are configured.${NC}"
fi

echo -e "${BLUE}Starting SocialRobot Voice Assistant...${NC}"
echo -e "Variant: ${GREEN}$VARIANT${NC}"
echo -e "Script:  ${GREEN}$SCRIPT${NC}"
echo -e "Log:     ${GREEN}$LOG_FILE${NC}"
if [ $# -gt 0 ]; then
    echo -e "Args:    ${GREEN}$@${NC}"
fi
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Start the bot in the background
nohup python3 "$SCRIPT" "$@" > "$LOG_FILE" 2>&1 &
BOT_PID=$!

# Save PID
echo "$BOT_PID" > "$PID_FILE"

# Wait a moment to check if it started successfully
sleep 2

if ps -p "$BOT_PID" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Bot started successfully with PID $BOT_PID${NC}"
    echo ""
    echo "To view logs:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To stop the bot:"
    echo "  ./stop_bot.sh"
else
    echo -e "${RED}✗ Bot failed to start. Check the logs:${NC}"
    echo "  cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

