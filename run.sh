#!/bin/bash
# Convenience script to run the SocialRobot voice assistant

echo "🤖 Starting SocialRobot Voice Assistant..."
echo "Using Python environment: socialrobot (Python 3.10)"
echo ""

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Deactivate any existing environment and activate socialrobot
conda deactivate 2>/dev/null || true
conda activate socialrobot

# Verify correct Python
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""

# Run the voice assistant
python main.py
