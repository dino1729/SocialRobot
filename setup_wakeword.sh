#!/bin/bash

# Setup script for Wake Word functionality
# This script installs the openwakeword package and downloads wake word models

set -e

echo "=================================================="
echo "Wake Word Setup for Social Robot"
echo "=================================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected!"
    echo "   Please activate your virtual environment first:"
    echo "   source ~/robot/.venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install openwakeword if not already installed
echo "-> Installing openwakeword..."
pip install openwakeword>=0.6.0

echo ""
echo "-> Testing wake word installation..."
python3 << 'EOF'
try:
    from openwakeword.model import Model
    print("✓ OpenWakeWord imported successfully")
    
    # Initialize with default model (will download if needed)
    print("\n-> Downloading default wake word model (hey_jarvis_v0.1)...")
    model = Model(wakeword_models=["hey_jarvis_v0.1"], inference_framework="onnx")
    print("✓ Wake word model loaded successfully")
    
    # List available models
    print("\n-> Available wake word models:")
    import os
    from pathlib import Path
    
    # Try to find the models directory
    try:
        import openwakeword
        oww_path = Path(openwakeword.__file__).parent
        models_path = oww_path / "models"
        
        if models_path.exists():
            models = [f.stem for f in models_path.glob("*.onnx") if f.is_file()]
            if models:
                for m in models:
                    print(f"   - {m}")
            else:
                print("   (No models found in default directory)")
        else:
            print("   (Models directory not found)")
    except Exception as e:
        print(f"   (Could not list models: {e})")
    
    print("\n✓ Wake word setup complete!")
    
except ImportError as e:
    print(f"✗ Failed to import openwakeword: {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error during setup: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ Wake Word Setup Complete!"
    echo "=================================================="
    echo ""
    echo "Available wake word models:"
    echo "  - hey_jarvis_v0.1 (default) - Say 'Hey Jarvis'"
    echo "  - alexa_v0.1 - Say 'Alexa'"
    echo "  - hey_mycroft_v0.1 - Say 'Hey Mycroft'"
    echo "  - hey_rhasspy_v0.1 - Say 'Hey Rhasspy'"
    echo ""
    echo "Configuration:"
    echo "  1. Copy env.example to .env if you haven't already"
    echo "  2. Edit .env and set:"
    echo "     WAKEWORD_MODEL=hey_jarvis_v0.1"
    echo "     WAKEWORD_THRESHOLD=0.5"
    echo ""
    echo "Usage:"
    echo "  Basic wake word assistant:"
    echo "    python main_wakeword.py"
    echo ""
    echo "  Internet-connected wake word assistant:"
    echo "    python main_wakeword_internetconnected.py"
    echo ""
    echo "  Custom threshold (0.0-1.0):"
    echo "    python main_wakeword.py --wakeword-threshold 0.6"
    echo ""
    echo "For more information, see: WAKEWORD_GUIDE.md"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "✗ Setup Failed"
    echo "=================================================="
    echo ""
    echo "Please check the error messages above and try again."
    echo "If the issue persists, try:"
    echo "  pip install --upgrade openwakeword"
    echo ""
    echo "Or install manually:"
    echo "  pip install openwakeword>=0.6.0"
    echo "=================================================="
    exit 1
fi

