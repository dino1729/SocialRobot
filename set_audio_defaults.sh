#!/bin/bash
# Script to set audio defaults for the voice assistant
# Run this if audio defaults reset after reboot

echo "Setting audio defaults for voice assistant..."
echo ""

# Check if PulseAudio is running and responsive
echo "Checking PulseAudio status..."
if ! pactl info &>/dev/null; then
    echo "⚠️  PulseAudio not responding. Restarting..."
    
    # Kill any existing PulseAudio instances
    pulseaudio --kill 2>/dev/null
    sleep 1
    
    # Start PulseAudio fresh
    pulseaudio --start 2>/dev/null
    sleep 2
    
    # Verify it's working
    if pactl info &>/dev/null; then
        echo "✓ PulseAudio started successfully"
    else
        echo "❌ ERROR: Failed to start PulseAudio"
        echo "   Try running manually: pulseaudio --kill && pulseaudio --start"
        exit 1
    fi
else
    echo "✓ PulseAudio is running"
fi
echo ""

# Set default input to USB microphone
pactl set-default-source alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-mono

# Set default output to USB speakers
pactl set-default-sink alsa_output.usb-Jieli_Technology_UACDemoV1.0_503581151344B11F-00.analog-stereo

echo ""
echo "=== Audio Defaults Set ==="
echo "Input (Mic): $(pactl get-default-source)"
echo "Output (Speaker): $(pactl get-default-sink)"
echo ""
echo "✓ Ready to run the voice assistant!"

