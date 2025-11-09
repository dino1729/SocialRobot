#!/bin/bash
# Source this file to activate the socialrobot conda environment
# Usage: source activate_env.sh

# Deactivate any existing conda environment first
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    conda deactivate 2>/dev/null || true
fi

# Initialize conda for the current shell
eval "$(conda shell.bash hook)"

# Activate the socialrobot environment
conda activate socialrobot

echo "✅ Conda environment 'socialrobot' activated!"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""
echo "Now you can run:"
echo "  python main.py"
echo ""
echo "To deactivate when done:"
echo "  conda deactivate"
