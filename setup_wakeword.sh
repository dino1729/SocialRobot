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
    # Try to find and activate a virtual environment
    VENV_FOUND=false
    
    # Check common venv locations
    for venv_path in ".venv" "venv" ".virtualenv" "env"; do
        if [[ -f "$venv_path/bin/activate" ]]; then
            echo "-> Found virtual environment at: $venv_path"
            echo "-> Activating virtual environment..."
            source "$venv_path/bin/activate"
            VENV_FOUND=true
            break
        fi
    done
    
    if [[ "$VENV_FOUND" == false ]]; then
        echo "⚠️  Warning: No virtual environment detected!"
        echo "   Please create a virtual environment first:"
        echo "   python3 -m venv .venv"
        echo "   source .venv/bin/activate"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ Virtual environment activated"
        echo ""
    fi
fi

# Install openwakeword if not already installed
echo "-> Installing openwakeword..."
# Using version 0.4.0 which includes ONNX models and doesn't require tflite-runtime
# Version 0.6.0 has dependency issues with tflite-runtime on some platforms
pip install openwakeword==0.4.0

echo ""
echo "-> Testing wake word installation..."
python3 << 'EOF'
try:
    from openwakeword.model import Model
    print("✓ OpenWakeWord imported successfully")
    
    # Initialize with default model (using 0.4.0 API with wakeword_model_paths)
    print("\n-> Loading default wake word model (hey_jarvis_v0.1)...")
    from pathlib import Path
    import openwakeword
    
    oww_path = Path(openwakeword.__file__).parent
    models_dir = oww_path / "resources" / "models"
    model_path = models_dir / "hey_jarvis_v0.1.onnx"
    
    if model_path.exists():
        model = Model(wakeword_model_paths=[str(model_path)])
        print("✓ Wake word model loaded successfully")
    else:
        print(f"✗ Model not found at {model_path}")
        exit(1)
    
    # List available models
    print("\n-> Available wake word models:")
    
    if models_dir.exists():
        models = [f.stem for f in models_dir.glob("*.onnx") if f.is_file() and not f.stem.startswith(('melspectrogram', 'embedding', 'silero'))]
        if models:
            for m in models:
                print(f"   - {m}")
        else:
            print("   (No wake word models found)")
    else:
        print("   (Models directory not found)")
    
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
    echo "  pip uninstall openwakeword"
    echo "  pip install openwakeword==0.4.0"
    echo ""
    echo "Note: Using version 0.4.0 which includes ONNX models"
    echo "      and doesn't require tflite-runtime"
    echo "=================================================="
    exit 1
fi

