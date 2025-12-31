#!/bin/bash
# Script to set audio defaults for the voice assistant
# Run this if audio defaults reset after reboot
# Features: Auto-detection, device testing, user selection, persistent config

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/.audio_config"

echo "============================================================"
echo "Audio Device Configuration"
echo "============================================================"
echo ""

# Parse arguments
AUTO_MODE=false
FORCE_RECONFIG=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --auto) AUTO_MODE=true; shift ;;
        --reset) FORCE_RECONFIG=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --auto    Auto-select devices without prompts"
            echo "  --reset   Ignore saved config and reconfigure"
            echo "  -h        Show this help"
            exit 0
            ;;
        *) shift ;;
    esac
done

# Check if PulseAudio/PipeWire is running and responsive
echo "Checking audio server status..."
if ! pactl info &>/dev/null; then
    echo "⚠️  Audio server not responding. Attempting restart..."
    
    # Try PipeWire first (modern systems), then PulseAudio
    if command -v pipewire &>/dev/null; then
        systemctl --user restart pipewire pipewire-pulse 2>/dev/null
    else
        pulseaudio --kill 2>/dev/null
        sleep 1
        pulseaudio --start 2>/dev/null
    fi
    sleep 2
    
    if pactl info &>/dev/null; then
        echo "✓ Audio server started successfully"
    else
        echo "❌ ERROR: Failed to start audio server"
        exit 1
    fi
else
    echo "✓ Audio server is running"
fi
echo ""

# Function to test if a sink (speaker) actually produces sound
test_sink() {
    local sink="$1"
    local duration="${2:-1}"
    
    # Try to play a short test tone
    pactl set-sink-mute "$sink" 0 2>/dev/null
    timeout "$duration" speaker-test -D "$sink" -t sine -f 440 -l 1 &>/dev/null &
    local pid=$!
    sleep "$duration"
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
    return 0
}

# Function to test if a source (microphone) can record
test_source() {
    local source="$1"
    local temp_file="/tmp/mic_test_$$.wav"
    
    # Try to record a short sample and check if we got audio data
    timeout 1 parecord --device="$source" --file-format=wav "$temp_file" 2>/dev/null
    
    if [[ -f "$temp_file" ]]; then
        local size=$(stat -f%z "$temp_file" 2>/dev/null || stat -c%s "$temp_file" 2>/dev/null)
        rm -f "$temp_file"
        # If file is larger than header (44 bytes), we got some audio
        [[ $size -gt 100 ]] && return 0
    fi
    return 1
}

# Function to get a friendly name for a device
get_device_description() {
    local device="$1"
    local type="$2"  # "sink" or "source"
    
    if [[ "$type" == "sink" ]]; then
        pactl list sinks | grep -A 50 "Name: $device" | grep "Description:" | head -1 | sed 's/.*Description: //'
    else
        pactl list sources | grep -A 50 "Name: $device" | grep "Description:" | head -1 | sed 's/.*Description: //'
    fi
}

# Function to check if device is USB
is_usb_device() {
    local device="$1"
    echo "$device" | grep -qi "usb"
}

# Function to get volume/gain percentage for a device
get_device_volume() {
    local device="$1"
    local type="$2"  # "sink" or "source"
    
    if [[ "$type" == "sink" ]]; then
        pactl get-sink-volume "$device" 2>/dev/null | grep -oP '\d+%' | head -1
    else
        pactl get-source-volume "$device" 2>/dev/null | grep -oP '\d+%' | head -1
    fi
}

# Function to get mute status for a device
get_device_mute() {
    local device="$1"
    local type="$2"  # "sink" or "source"
    
    if [[ "$type" == "sink" ]]; then
        pactl get-sink-mute "$device" 2>/dev/null | grep -oP '(yes|no)'
    else
        pactl get-source-mute "$device" 2>/dev/null | grep -oP '(yes|no)'
    fi
}

# Function to display volume bar
display_volume_bar() {
    local percent="$1"
    local width=20
    local num="${percent%\%}"
    local filled=$((num * width / 100))
    local empty=$((width - filled))
    
    printf "["
    for ((j=0; j<filled; j++)); do printf "#"; done
    for ((j=0; j<empty; j++)); do printf "-"; done
    printf "] %s" "$percent"
}

# Function to score a device (higher is better)
score_device() {
    local device="$1"
    local score=0
    
    # Prefer GeneralPlus (known working USB audio chips)
    echo "$device" | grep -qi "generalplus" && ((score += 100))
    
    # Prefer USB devices
    is_usb_device "$device" && ((score += 50))
    
    # Prefer analog over digital/iec958 (usually more compatible)
    echo "$device" | grep -qi "analog" && ((score += 20))
    
    # Deprioritize monitors (they're not real microphones)
    echo "$device" | grep -qi "monitor" && ((score -= 200))
    
    # Deprioritize HDMI (usually for video output, not speakers)
    echo "$device" | grep -qi "hdmi" && ((score -= 50))
    
    # Deprioritize iec958/spdif (digital, often doesn't work without proper receiver)
    echo "$device" | grep -qi "iec958\|spdif" && ((score -= 30))
    
    echo $score
}

# Collect all available sinks (speakers)
echo "=== Scanning Audio Devices ==="
echo ""

declare -a SINKS
declare -a SINK_NAMES
declare -a SINK_SCORES
echo "Available Output Devices (Speakers):"
echo "-------------------------------------"
i=0
while IFS=$'\t' read -r id name driver format state; do
    # Skip monitors
    if echo "$name" | grep -qi "monitor"; then
        continue
    fi
    desc=$(get_device_description "$name" "sink")
    score=$(score_device "$name")
    SINKS[$i]="$name"
    SINK_NAMES[$i]="$desc"
    SINK_SCORES[$i]=$score
    
    usb_marker=""
    is_usb_device "$name" && usb_marker="[USB] "
    echo "  $((i+1)). ${usb_marker}${desc:-$name}"
    echo "      Device: $name"
    echo "      Score: $score"
    ((i++))
done < <(pactl list sinks short)
SINK_COUNT=$i
echo ""

# Collect all available sources (microphones)
declare -a SOURCES
declare -a SOURCE_NAMES
declare -a SOURCE_SCORES
echo "Available Input Devices (Microphones):"
echo "---------------------------------------"
i=0
while IFS=$'\t' read -r id name driver format state; do
    # Skip monitors (they capture system audio, not microphone)
    if echo "$name" | grep -qi "monitor"; then
        continue
    fi
    desc=$(get_device_description "$name" "source")
    score=$(score_device "$name")
    SOURCES[$i]="$name"
    SOURCE_NAMES[$i]="$desc"
    SOURCE_SCORES[$i]=$score
    
    usb_marker=""
    is_usb_device "$name" && usb_marker="[USB] "
    echo "  $((i+1)). ${usb_marker}${desc:-$name}"
    echo "      Device: $name"
    echo "      Score: $score"
    ((i++))
done < <(pactl list sources short)
SOURCE_COUNT=$i
echo ""

# Check if we have saved config and devices still exist
SAVED_SOURCE=""
SAVED_SINK=""
if [[ -f "$CONFIG_FILE" ]] && [[ "$FORCE_RECONFIG" == "false" ]]; then
    source "$CONFIG_FILE"
    
    # Verify saved devices still exist
    if [[ -n "$SAVED_SOURCE" ]]; then
        if ! pactl list sources short | grep -q "$SAVED_SOURCE"; then
            echo "⚠️  Previously saved microphone no longer available"
            SAVED_SOURCE=""
        fi
    fi
    if [[ -n "$SAVED_SINK" ]]; then
        if ! pactl list sinks short | grep -q "$SAVED_SINK"; then
            echo "⚠️  Previously saved speaker no longer available"
            SAVED_SINK=""
        fi
    fi
    
    if [[ -n "$SAVED_SOURCE" ]] && [[ -n "$SAVED_SINK" ]]; then
        echo "Found saved configuration:"
        echo "  Microphone: $(get_device_description "$SAVED_SOURCE" "source")"
        echo "  Speaker:    $(get_device_description "$SAVED_SINK" "sink")"
        echo ""
        
        if [[ "$AUTO_MODE" == "true" ]]; then
            USE_SAVED="y"
        else
            read -p "Use saved configuration? (Y/n) " -n 1 -r USE_SAVED
            echo
        fi
        
        if [[ ! $USE_SAVED =~ ^[Nn]$ ]]; then
            pactl set-default-source "$SAVED_SOURCE"
            pactl set-default-sink "$SAVED_SINK"
            echo ""
            echo "✓ Applied saved configuration"
            echo ""
            echo "=== Current Audio Defaults ==="
            echo "Input (Mic):     $(pactl get-default-source)"
            echo "Output (Speaker): $(pactl get-default-sink)"
            echo ""
            
            # Display volume/gain levels
            echo "=== Volume / Gain Levels ==="
            SINK_VOL=$(get_device_volume "$SAVED_SINK" "sink")
            SINK_MUTE=$(get_device_mute "$SAVED_SINK" "sink")
            echo -n "Speaker Volume:    "
            display_volume_bar "$SINK_VOL"
            if [[ "$SINK_MUTE" == "yes" ]]; then
                echo " [MUTED]"
            else
                echo ""
            fi
            
            SOURCE_VOL=$(get_device_volume "$SAVED_SOURCE" "source")
            SOURCE_MUTE=$(get_device_mute "$SAVED_SOURCE" "source")
            echo -n "Microphone Gain:   "
            display_volume_bar "$SOURCE_VOL"
            if [[ "$SOURCE_MUTE" == "yes" ]]; then
                echo " [MUTED]"
            else
                echo ""
            fi
            echo ""
            
            # Warn if volumes are too low or muted
            if [[ "$SINK_MUTE" == "yes" ]]; then
                echo "⚠️  Warning: Speaker is MUTED! Run: pactl set-sink-mute $SAVED_SINK 0"
            elif [[ -n "$SINK_VOL" ]] && [[ "${SINK_VOL%\%}" -lt 20 ]]; then
                echo "⚠️  Warning: Speaker volume is very low ($SINK_VOL)"
            fi
            
            if [[ "$SOURCE_MUTE" == "yes" ]]; then
                echo "⚠️  Warning: Microphone is MUTED! Run: pactl set-source-mute $SAVED_SOURCE 0"
            elif [[ -n "$SOURCE_VOL" ]] && [[ "${SOURCE_VOL%\%}" -lt 20 ]]; then
                echo "⚠️  Warning: Microphone gain is very low ($SOURCE_VOL)"
            fi
            
            echo ""
            echo "✅ Audio configured successfully!"
            exit 0
        fi
        echo ""
    fi
fi

# Auto-select best devices based on score
echo "=== Selecting Best Devices ==="
echo ""

# Find best sink
BEST_SINK=""
BEST_SINK_SCORE=-999
for ((i=0; i<SINK_COUNT; i++)); do
    if [[ ${SINK_SCORES[$i]} -gt $BEST_SINK_SCORE ]]; then
        BEST_SINK_SCORE=${SINK_SCORES[$i]}
        BEST_SINK="${SINKS[$i]}"
        BEST_SINK_NAME="${SINK_NAMES[$i]}"
    fi
done

# Find best source
BEST_SOURCE=""
BEST_SOURCE_SCORE=-999
for ((i=0; i<SOURCE_COUNT; i++)); do
    if [[ ${SOURCE_SCORES[$i]} -gt $BEST_SOURCE_SCORE ]]; then
        BEST_SOURCE_SCORE=${SOURCE_SCORES[$i]}
        BEST_SOURCE="${SOURCES[$i]}"
        BEST_SOURCE_NAME="${SOURCE_NAMES[$i]}"
    fi
done

echo "Recommended devices (based on scoring):"
echo "  Microphone: ${BEST_SOURCE_NAME:-None found}"
echo "  Speaker:    ${BEST_SINK_NAME:-None found}"
echo ""

# In auto mode, just use the best devices
if [[ "$AUTO_MODE" == "true" ]]; then
    SELECTED_SOURCE="$BEST_SOURCE"
    SELECTED_SINK="$BEST_SINK"
else
    # Let user confirm or choose different devices
    read -p "Use recommended devices? (Y/n) " -n 1 -r USE_RECOMMENDED
    echo
    
    if [[ $USE_RECOMMENDED =~ ^[Nn]$ ]]; then
        # Manual selection for sink
        echo ""
        echo "Select output device (speaker):"
        for ((i=0; i<SINK_COUNT; i++)); do
            echo "  $((i+1)). ${SINK_NAMES[$i]:-${SINKS[$i]}}"
        done
        read -p "Enter number (1-$SINK_COUNT): " SINK_CHOICE
        if [[ $SINK_CHOICE =~ ^[0-9]+$ ]] && [[ $SINK_CHOICE -ge 1 ]] && [[ $SINK_CHOICE -le $SINK_COUNT ]]; then
            SELECTED_SINK="${SINKS[$((SINK_CHOICE-1))]}"
        else
            echo "Invalid selection, using recommended"
            SELECTED_SINK="$BEST_SINK"
        fi
        
        # Manual selection for source
        echo ""
        echo "Select input device (microphone):"
        for ((i=0; i<SOURCE_COUNT; i++)); do
            echo "  $((i+1)). ${SOURCE_NAMES[$i]:-${SOURCES[$i]}}"
        done
        read -p "Enter number (1-$SOURCE_COUNT): " SOURCE_CHOICE
        if [[ $SOURCE_CHOICE =~ ^[0-9]+$ ]] && [[ $SOURCE_CHOICE -ge 1 ]] && [[ $SOURCE_CHOICE -le $SOURCE_COUNT ]]; then
            SELECTED_SOURCE="${SOURCES[$((SOURCE_CHOICE-1))]}"
        else
            echo "Invalid selection, using recommended"
            SELECTED_SOURCE="$BEST_SOURCE"
        fi
    else
        SELECTED_SOURCE="$BEST_SOURCE"
        SELECTED_SINK="$BEST_SINK"
    fi
fi

# Apply the configuration
echo ""
echo "=== Applying Configuration ==="

if [[ -z "$SELECTED_SOURCE" ]]; then
    echo "❌ ERROR: No microphone available"
    SOURCE_OK=false
else
    pactl set-default-source "$SELECTED_SOURCE"
    echo "✓ Set microphone: $(get_device_description "$SELECTED_SOURCE" "source")"
    SOURCE_OK=true
fi

if [[ -z "$SELECTED_SINK" ]]; then
    echo "❌ ERROR: No speaker available"
    SINK_OK=false
else
    pactl set-default-sink "$SELECTED_SINK"
    echo "✓ Set speaker: $(get_device_description "$SELECTED_SINK" "sink")"
    SINK_OK=true
fi

echo ""

# Test the devices
if [[ "$AUTO_MODE" == "false" ]] && $SOURCE_OK && $SINK_OK; then
    read -p "Would you like to test the audio devices? (Y/n) " -n 1 -r TEST_AUDIO
    echo
    
    if [[ ! $TEST_AUDIO =~ ^[Nn]$ ]]; then
        echo ""
        echo "=== Testing Speaker ==="
        echo "Playing test tone for 2 seconds..."
        test_sink "$SELECTED_SINK" 2
        
        read -p "Did you hear the tone? (Y/n) " -n 1 -r HEARD_TONE
        echo
        
        if [[ $HEARD_TONE =~ ^[Nn]$ ]]; then
            echo ""
            echo "⚠️  Speaker test failed. You may want to:"
            echo "   - Check speaker volume and connections"
            echo "   - Run this script again with --reset to choose a different device"
            SINK_OK=false
        else
            echo "✓ Speaker test passed"
        fi
        
        echo ""
        echo "=== Testing Microphone ==="
        echo "Recording 3 seconds of audio. Please speak into the microphone..."
        
        TEMP_AUDIO="/tmp/mic_test_$$.wav"
        timeout 3 parecord --device="$SELECTED_SOURCE" --file-format=wav "$TEMP_AUDIO" 2>/dev/null
        
        echo "Playing back recording..."
        paplay "$TEMP_AUDIO" 2>/dev/null
        rm -f "$TEMP_AUDIO"
        
        read -p "Did you hear your voice? (Y/n) " -n 1 -r HEARD_VOICE
        echo
        
        if [[ $HEARD_VOICE =~ ^[Nn]$ ]]; then
            echo ""
            echo "⚠️  Microphone test failed. You may want to:"
            echo "   - Check microphone connections"
            echo "   - Ensure microphone is not muted in system settings"
            echo "   - Run this script again with --reset to choose a different device"
            SOURCE_OK=false
        else
            echo "✓ Microphone test passed"
        fi
    fi
fi

echo ""
echo "=== Final Configuration ==="
echo "Input (Mic):     $(pactl get-default-source)"
echo "Output (Speaker): $(pactl get-default-sink)"
echo ""

# Display volume/gain levels
echo "=== Volume / Gain Levels ==="
CURRENT_SOURCE=$(pactl get-default-source)
CURRENT_SINK=$(pactl get-default-sink)

# Speaker volume
SINK_VOL=$(get_device_volume "$CURRENT_SINK" "sink")
SINK_MUTE=$(get_device_mute "$CURRENT_SINK" "sink")
echo -n "Speaker Volume:    "
display_volume_bar "$SINK_VOL"
if [[ "$SINK_MUTE" == "yes" ]]; then
    echo " [MUTED]"
else
    echo ""
fi

# Microphone gain
SOURCE_VOL=$(get_device_volume "$CURRENT_SOURCE" "source")
SOURCE_MUTE=$(get_device_mute "$CURRENT_SOURCE" "source")
echo -n "Microphone Gain:   "
display_volume_bar "$SOURCE_VOL"
if [[ "$SOURCE_MUTE" == "yes" ]]; then
    echo " [MUTED]"
else
    echo ""
fi

echo ""

# Warn if volumes are too low or muted
if [[ "$SINK_MUTE" == "yes" ]]; then
    echo "⚠️  Warning: Speaker is MUTED! Run: pactl set-sink-mute $CURRENT_SINK 0"
elif [[ "${SINK_VOL%\%}" -lt 20 ]]; then
    echo "⚠️  Warning: Speaker volume is very low ($SINK_VOL)"
fi

if [[ "$SOURCE_MUTE" == "yes" ]]; then
    echo "⚠️  Warning: Microphone is MUTED! Run: pactl set-source-mute $CURRENT_SOURCE 0"
elif [[ "${SOURCE_VOL%\%}" -lt 20 ]]; then
    echo "⚠️  Warning: Microphone gain is very low ($SOURCE_VOL)"
fi

echo ""

# Save configuration if tests passed
if $SOURCE_OK && $SINK_OK; then
    echo "# Audio configuration saved on $(date)" > "$CONFIG_FILE"
    echo "SAVED_SOURCE=\"$SELECTED_SOURCE\"" >> "$CONFIG_FILE"
    echo "SAVED_SINK=\"$SELECTED_SINK\"" >> "$CONFIG_FILE"
    echo "✓ Configuration saved to $CONFIG_FILE"
    echo ""
    echo "============================================================"
    echo "✅ Audio devices configured successfully!"
    echo "============================================================"
    echo ""
    echo "You can now run the voice assistant:"
    echo "  source .venv/bin/activate && python main.py"
    echo ""
    echo "Tips:"
    echo "  - Run with --auto for non-interactive setup"
    echo "  - Run with --reset to reconfigure devices"
    exit 0
else
    echo "============================================================"
    echo "⚠️  Configuration incomplete or tests failed"
    echo "============================================================"
    echo ""
    echo "Please check your audio devices and try again."
    echo "Run: $0 --reset"
    exit 1
fi

