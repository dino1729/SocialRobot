#!/bin/bash
# Script to check audio defaults before running the voice assistant
# This helps verify that audio devices are properly configured

echo "============================================================"
echo "Audio Configuration Check"
echo "============================================================"
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

# Get current defaults
CURRENT_SOURCE=$(pactl get-default-source 2>/dev/null)
CURRENT_SINK=$(pactl get-default-sink 2>/dev/null)

echo "=== Current Audio Defaults ==="
echo "Input (Mic):     $CURRENT_SOURCE"
echo "Output (Speaker): $CURRENT_SINK"
echo ""

# Auto-detect USB audio devices (prefer GeneralPlus over generic USB)
echo "-> Detecting USB audio devices..."
# First try to find GeneralPlus USB devices, then fall back to any USB device
EXPECTED_SOURCE=$(pactl list sources short | grep -i "generalplus" | grep -v "monitor" | head -n1 | awk '{print $2}' || true)
if [[ -z "$EXPECTED_SOURCE" ]]; then
    EXPECTED_SOURCE=$(pactl list sources short | grep -i "usb" | grep -v "monitor" | head -n1 | awk '{print $2}' || true)
fi
EXPECTED_SINK=$(pactl list sinks short | grep -i "generalplus" | head -n1 | awk '{print $2}' || true)
if [[ -z "$EXPECTED_SINK" ]]; then
    EXPECTED_SINK=$(pactl list sinks short | grep -i "usb" | head -n1 | awk '{print $2}' || true)
fi

if [[ -z "$EXPECTED_SOURCE" ]]; then
    echo "⚠️  No USB microphone detected"
    EXPECTED_SOURCE="none"
else
    echo "✓ Found USB microphone: $EXPECTED_SOURCE"
fi

if [[ -z "$EXPECTED_SINK" ]]; then
    echo "⚠️  No USB speaker detected"
    EXPECTED_SINK="none"
else
    echo "✓ Found USB speaker: $EXPECTED_SINK"
fi
echo ""

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
if [[ "$EXPECTED_SOURCE" == "none" ]]; then
    echo "⚠️  Microphone:  No USB device available"
    echo "  Current: $CURRENT_SOURCE"
elif $SOURCE_OK; then
    echo "✓ Microphone:  Using USB device (CORRECT)"
else
    echo "✗ Microphone:  NOT using USB device"
    echo "  Current: $CURRENT_SOURCE"
    echo "  Expected: $EXPECTED_SOURCE"
fi

if [[ "$EXPECTED_SINK" == "none" ]]; then
    echo "⚠️  Speaker:     No USB device available"
    echo "  Current: $CURRENT_SINK"
elif $SINK_OK; then
    echo "✓ Speaker:     Using USB device (CORRECT)"
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
    
    # Offer audio test
    read -p "Would you like to test speaker and microphone? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo "=== Audio Hardware Test ==="
        echo ""
        
        # Test speaker
        echo "-> Testing speaker (playing 2-second test tone)..."
        echo "   You should hear a tone from your USB speaker."
        paplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null || \
            speaker-test -t sine -f 1000 -l 1 -D default 2>/dev/null &
        SPEAKER_TEST_PID=$!
        sleep 2
        kill $SPEAKER_TEST_PID 2>/dev/null
        wait $SPEAKER_TEST_PID 2>/dev/null
        echo "   ✓ Speaker test complete"
        echo ""
        
        # Test microphone
        echo "-> Testing microphone..."
        echo "   Recording 5 seconds of audio. Please speak now!"
        TEMP_AUDIO="/tmp/mic_test_$$.wav"
        timeout 5 arecord -D default -f cd -t wav "$TEMP_AUDIO" 2>/dev/null
        echo "   ✓ Recording complete"
        echo ""
        
        echo "-> Playing back recording..."
        echo "   You should hear your voice from the speaker."
        paplay "$TEMP_AUDIO" 2>/dev/null || aplay "$TEMP_AUDIO" 2>/dev/null
        echo "   ✓ Playback complete"
        rm -f "$TEMP_AUDIO"
        echo ""
        
        # Ask for confirmation
        read -p "Did you hear the test tone and your voice? (Y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo ""
            echo "⚠️  Audio test failed. Please check:"
            echo "   - USB devices are properly connected"
            echo "   - Speaker volume is turned up"
            echo "   - Microphone is not muted"
            echo ""
            exit 1
        else
            echo ""
            echo "✅ Audio test successful!"
        fi
    fi
    
    echo ""
    echo "You can now run the voice assistant:"
    echo "  source .venv/bin/activate && python main.py"
    exit 0
else
    if [[ "$EXPECTED_SOURCE" == "none" ]] || [[ "$EXPECTED_SINK" == "none" ]]; then
        echo "⚠️  WARNING: USB audio devices not detected!"
        echo ""
        echo "Please connect USB microphone and/or speaker, then run:"
        echo "  ./check_audio_defaults.sh"
    else
        echo "⚠️  WARNING: Audio devices need to be configured!"
        echo ""
        echo "Run this command to fix:"
        echo "  ./set_audio_defaults.sh"
        echo ""
        echo "Or set them manually with pactl"
    fi
    exit 1
fi

