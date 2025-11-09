#!/bin/bash
# Script to check audio defaults before running the voice assistant
# This helps verify that audio devices are properly configured

echo "============================================================"
echo "Audio Configuration Check"
echo "============================================================"
echo ""

# Check if PulseAudio is running
if ! pulseaudio --check 2>/dev/null; then
    echo "⚠️  WARNING: PulseAudio is not running!"
    echo "   Starting PulseAudio..."
    pulseaudio --start
    sleep 1
fi

# Get current defaults
CURRENT_SOURCE=$(pactl get-default-source 2>/dev/null)
CURRENT_SINK=$(pactl get-default-sink 2>/dev/null)

echo "=== Current Audio Defaults ==="
echo "Input (Mic):     $CURRENT_SOURCE"
echo "Output (Speaker): $CURRENT_SINK"
echo ""

# Expected defaults (USB devices)
EXPECTED_SOURCE="alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-mono"
EXPECTED_SINK="alsa_output.usb-Jieli_Technology_UACDemoV1.0_503581151344B11F-00.analog-stereo"

# Check if devices match expected
SOURCE_OK=false
SINK_OK=false

if [[ "$CURRENT_SOURCE" == "$EXPECTED_SOURCE" ]]; then
    SOURCE_OK=true
fi

if [[ "$CURRENT_SINK" == "$EXPECTED_SINK" ]]; then
    SINK_OK=true
fi

# Show available devices
echo "=== Available Input Devices (Microphones) ==="
arecord -l 2>/dev/null | grep "^card" | head -5
echo ""

echo "=== Available Output Devices (Speakers) ==="
aplay -l 2>/dev/null | grep "^card" | head -5
echo ""

# Status check
echo "=== Configuration Status ==="
if $SOURCE_OK; then
    echo "✓ Microphone:  USB PnP Sound Device (CORRECT)"
else
    echo "✗ Microphone:  NOT using USB device"
    echo "  Current: $CURRENT_SOURCE"
    echo "  Expected: $EXPECTED_SOURCE"
fi

if $SINK_OK; then
    echo "✓ Speaker:     USB Audio (CORRECT)"
else
    echo "✗ Speaker:     NOT using USB device"
    echo "  Current: $CURRENT_SINK"
    echo "  Expected: $EXPECTED_SINK"
fi

echo ""
echo "============================================================"

# Final verdict
if $SOURCE_OK && $SINK_OK; then
    echo "✅ READY: Audio devices are properly configured!"
    echo ""
    echo "You can now run the voice assistant:"
    echo "  source venv/bin/activate && python main.py"
    exit 0
else
    echo "⚠️  WARNING: Audio devices need to be configured!"
    echo ""
    echo "Run this command to fix:"
    echo "  ./set_audio_defaults.sh"
    echo ""
    echo "Or set them manually with pactl"
    exit 1
fi

