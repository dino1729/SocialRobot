#!/bin/bash
# Script to start the SocialRobot voice assistant in the background
# Usage: ./start_bot.sh [VARIANT] [ADDITIONAL_ARGS...]
#
# VARIANT options:
#   basic                         - Continuous listening (Ollama) [DEFAULT]
#   wakeword                      - Wake word activated (Ollama)
#   wakeword_online               - Wake word activated (LiteLLM/OpenAI)
#   tools                         - Continuous with internet tools (Ollama)
#   wakeword_tools                - Wake word with internet tools (Ollama)
#   wakeword_tools_online         - Wake word with internet tools (LiteLLM/OpenAI)
#
# Examples:
#   ./start_bot.sh                                    # Start default (basic)
#   ./start_bot.sh wakeword                           # Start with wake word
#   ./start_bot.sh wakeword --tts-engine chatterbox --voice rick_sanchez --tts-gpu
#   ./start_bot.sh tools --compute-type int8

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
VARIANT="${1:-basic}"

# Check if first argument is a known variant and shift it
MAIN_ARGS=""
case "$1" in
    basic|wakeword|wakeword_online|tools|wakeword_tools|wakeword_tools_online)
        shift
        ;;
    # Also support legacy variant names
    wakeword_internetconnected)
        VARIANT="wakeword_tools"
        shift
        ;;
    wakeword_internetconnected_online)
        VARIANT="wakeword_tools_online"
        shift
        ;;
    *)
        # First arg is not a variant, use default
        VARIANT="basic"
        ;;
esac

# Map variant to main.py arguments
case "$VARIANT" in
    basic)
        MAIN_ARGS=""
        DESCRIPTION="Continuous listening (Ollama)"
        ;;
    wakeword)
        MAIN_ARGS="--wakeword"
        DESCRIPTION="Wake word activated (Ollama)"
        ;;
    wakeword_online)
        MAIN_ARGS="--wakeword --llm litellm"
        DESCRIPTION="Wake word activated (LiteLLM/OpenAI)"
        ;;
    tools)
        MAIN_ARGS="--tools"
        DESCRIPTION="Continuous with internet tools (Ollama)"
        ;;
    wakeword_tools)
        MAIN_ARGS="--wakeword --tools"
        DESCRIPTION="Wake word with internet tools (Ollama)"
        ;;
    wakeword_tools_online)
        MAIN_ARGS="--wakeword --llm litellm --tools"
        DESCRIPTION="Wake word with internet tools (LiteLLM/OpenAI)"
        ;;
    *)
        echo -e "${RED}Error: Unknown variant '$VARIANT'${NC}"
        echo ""
        echo "Available variants:"
        echo "  basic                 - Continuous listening (Ollama) [DEFAULT]"
        echo "  wakeword              - Wake word activated (Ollama)"
        echo "  wakeword_online       - Wake word activated (LiteLLM/OpenAI)"
        echo "  tools                 - Continuous with internet tools (Ollama)"
        echo "  wakeword_tools        - Wake word with internet tools (Ollama)"
        echo "  wakeword_tools_online - Wake word with internet tools (LiteLLM/OpenAI)"
        echo ""
        echo "Usage: ./start_bot.sh [VARIANT] [ADDITIONAL_ARGS...]"
        echo ""
        echo "Examples:"
        echo "  ./start_bot.sh wakeword --tts-engine chatterbox --voice rick_sanchez --tts-gpu"
        echo "  ./start_bot.sh tools --device cpu"
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

# Check if main.py exists
if [ ! -f "$SCRIPT_DIR/main.py" ]; then
    echo -e "${RED}Error: main.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Make sure environment variables are configured.${NC}"
fi

# Check audio configuration (PulseAudio)
echo -e "${BLUE}Checking audio configuration...${NC}"
if ! command -v pactl &> /dev/null; then
    echo -e "${YELLOW}Warning: pactl not found. Skipping audio check.${NC}"
else
    # Check if PulseAudio is running
    if ! pactl info &>/dev/null; then
        echo -e "${YELLOW}PulseAudio not responding. Attempting to restart...${NC}"
        pulseaudio --kill 2>/dev/null
        sleep 1
        pulseaudio --start 2>/dev/null
        sleep 2
        
        if ! pactl info &>/dev/null; then
            echo -e "${RED}Warning: PulseAudio failed to start. Audio may not work properly.${NC}"
            echo -e "${YELLOW}Try running: pulseaudio --kill && pulseaudio --start${NC}"
        else
            echo -e "${GREEN}✓ PulseAudio started${NC}"
        fi
    else
        echo -e "${GREEN}✓ PulseAudio is running${NC}"
    fi
    
    # Check audio devices
    CURRENT_SOURCE=$(pactl get-default-source 2>/dev/null)
    CURRENT_SINK=$(pactl get-default-sink 2>/dev/null)
    
    if [ -n "$CURRENT_SOURCE" ] && [ -n "$CURRENT_SINK" ]; then
        echo -e "${GREEN}✓ Audio devices configured${NC}"
        echo -e "  Input:  ${CURRENT_SOURCE}"
        echo -e "  Output: ${CURRENT_SINK}"
    else
        echo -e "${YELLOW}Warning: Audio devices not properly configured${NC}"
        echo -e "${YELLOW}Run ./check_audio_defaults.sh for details${NC}"
    fi
fi
echo ""

echo -e "${BLUE}Starting SocialRobot Voice Assistant...${NC}"
echo -e "Mode:    ${GREEN}$DESCRIPTION${NC}"
echo -e "Log:     ${GREEN}$LOG_FILE${NC}"
if [ -n "$MAIN_ARGS" ] || [ $# -gt 0 ]; then
    echo -e "Args:    ${GREEN}$MAIN_ARGS $@${NC}"
fi
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Set up CUDA library paths for GPU support (fixes cuDNN loading issues)
if [ -d "$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cudnn/lib" ]; then
    export LD_LIBRARY_PATH="$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
fi

# Activate virtual environment if it exists
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Start the bot in the background
nohup python3 main.py $MAIN_ARGS "$@" > "$LOG_FILE" 2>&1 &
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
