#!/bin/bash
# Script to set audio defaults for the voice assistant
# Run this if audio defaults reset after reboot

echo "Setting audio defaults for voice assistant..."

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

