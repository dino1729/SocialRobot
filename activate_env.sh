#!/bin/bash
# Source this file to activate the socialrobot virtual environment
# Usage: source activate_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "❌ Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Please create it first with: python -m venv .venv"
    return 1 2>/dev/null || exit 1
fi

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

echo "✅ Virtual environment activated!"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""
echo "Now you can run:"
echo "  python main.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"

