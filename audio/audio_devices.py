"""Audio device detection and configuration utilities.

This module provides centralized audio device management to ensure consistent
device selection across all components (TTS, STT, VAD, Wake Word).

It reads the saved configuration from .audio_config if available, and validates
that the system is using the correct default devices.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class AudioDeviceInfo:
    """Information about an audio device."""
    name: str  # PulseAudio/PipeWire device name
    description: str  # Human-readable description
    is_usb: bool  # Whether it's a USB device
    is_generalplus: bool  # Whether it's a GeneralPlus USB device (preferred)


def get_script_dir() -> Path:
    """Get the SocialRobot script directory."""
    # Navigate up from audio/ to the project root
    return Path(__file__).parent.parent


def get_saved_config() -> Tuple[Optional[str], Optional[str]]:
    """Read saved audio configuration.
    
    Returns:
        Tuple of (saved_source, saved_sink) or (None, None) if not configured
    """
    config_file = get_script_dir() / ".audio_config"
    
    if not config_file.exists():
        return None, None
    
    saved_source = None
    saved_sink = None
    
    try:
        with open(config_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("SAVED_SOURCE="):
                    saved_source = line.split("=", 1)[1].strip('"')
                elif line.startswith("SAVED_SINK="):
                    saved_sink = line.split("=", 1)[1].strip('"')
    except Exception:
        pass
    
    return saved_source, saved_sink


def get_current_defaults() -> Tuple[Optional[str], Optional[str]]:
    """Get current PulseAudio/PipeWire default devices.
    
    Returns:
        Tuple of (default_source, default_sink)
    """
    try:
        source = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True, text=True, timeout=5
        )
        sink = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, timeout=5
        )
        return (
            source.stdout.strip() if source.returncode == 0 else None,
            sink.stdout.strip() if sink.returncode == 0 else None
        )
    except Exception:
        return None, None


def device_exists(device_name: str, device_type: str = "source") -> bool:
    """Check if a PulseAudio device exists.
    
    Args:
        device_name: PulseAudio device name
        device_type: "source" for microphones, "sink" for speakers
        
    Returns:
        True if device exists
    """
    try:
        cmd = ["pactl", "list", f"{device_type}s", "short"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return device_name in result.stdout
    except Exception:
        return False


def set_default_device(device_name: str, device_type: str = "source") -> bool:
    """Set the default audio device.
    
    Args:
        device_name: PulseAudio device name
        device_type: "source" for microphones, "sink" for speakers
        
    Returns:
        True if successful
    """
    try:
        cmd = ["pactl", f"set-default-{device_type}", device_name]
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def ensure_correct_audio_devices(verbose: bool = True) -> bool:
    """Ensure the correct audio devices are set as defaults.
    
    This function:
    1. Reads the saved configuration from .audio_config
    2. Checks if current defaults match the saved config
    3. If not, corrects them automatically
    
    Args:
        verbose: Print status messages
        
    Returns:
        True if audio devices are correctly configured
    """
    saved_source, saved_sink = get_saved_config()
    
    if not saved_source or not saved_sink:
        if verbose:
            print("-> No saved audio configuration found.")
            print("   Run ./set_audio_defaults.sh to configure audio devices.")
        return False
    
    current_source, current_sink = get_current_defaults()
    
    source_ok = current_source == saved_source
    sink_ok = current_sink == saved_sink
    
    if source_ok and sink_ok:
        if verbose:
            print("-> Audio devices correctly configured:")
            print(f"   Microphone: {saved_source}")
            print(f"   Speaker: {saved_sink}")
        return True
    
    # Attempt to fix incorrect settings
    if verbose:
        print("-> Correcting audio device configuration...")
    
    fixed = True
    
    if not source_ok:
        if device_exists(saved_source, "source"):
            if set_default_device(saved_source, "source"):
                if verbose:
                    print(f"   ✓ Set microphone to: {saved_source}")
            else:
                if verbose:
                    print(f"   ✗ Failed to set microphone to: {saved_source}")
                fixed = False
        else:
            if verbose:
                print(f"   ✗ Saved microphone device not found: {saved_source}")
            fixed = False
    
    if not sink_ok:
        if device_exists(saved_sink, "sink"):
            if set_default_device(saved_sink, "sink"):
                if verbose:
                    print(f"   ✓ Set speaker to: {saved_sink}")
            else:
                if verbose:
                    print(f"   ✗ Failed to set speaker to: {saved_sink}")
                fixed = False
        else:
            if verbose:
                print(f"   ✗ Saved speaker device not found: {saved_sink}")
            fixed = False
    
    return fixed


def get_pyaudio_device_index(
    device_name: str,
    is_input: bool = True
) -> Optional[int]:
    """Get PyAudio device index for a PulseAudio device name.
    
    Note: PyAudio uses the system default when device_index=None,
    so this is typically not needed if system defaults are correct.
    
    Args:
        device_name: Part of the device name to match
        is_input: True for input devices, False for output
        
    Returns:
        PyAudio device index or None if not found
    """
    try:
        from audio.suppress_warnings import get_pyaudio, suppress_stderr
        pyaudio = get_pyaudio()
        with suppress_stderr():
            pa = pyaudio.PyAudio()
        
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info.get("name", "")
            
            # Check if this device matches and has the right channels
            if device_name.lower() in name.lower():
                if is_input and info.get("maxInputChannels", 0) > 0:
                    pa.terminate()
                    return i
                elif not is_input and info.get("maxOutputChannels", 0) > 0:
                    pa.terminate()
                    return i
        
        pa.terminate()
    except Exception:
        pass
    
    return None


def validate_audio_setup() -> dict:
    """Validate the complete audio setup.
    
    Returns:
        Dict with validation results:
        {
            "pulseaudio_running": bool,
            "saved_config_exists": bool,
            "source_configured": bool,
            "sink_configured": bool,
            "source_name": str or None,
            "sink_name": str or None,
            "all_ok": bool,
            "issues": list[str]
        }
    """
    result = {
        "pulseaudio_running": False,
        "saved_config_exists": False,
        "source_configured": False,
        "sink_configured": False,
        "source_name": None,
        "sink_name": None,
        "all_ok": False,
        "issues": []
    }
    
    # Check PulseAudio
    try:
        proc = subprocess.run(
            ["pactl", "info"],
            capture_output=True, timeout=5
        )
        result["pulseaudio_running"] = proc.returncode == 0
    except Exception:
        result["issues"].append("Cannot communicate with PulseAudio/PipeWire")
        return result
    
    if not result["pulseaudio_running"]:
        result["issues"].append("PulseAudio/PipeWire is not running")
        return result
    
    # Check saved config
    saved_source, saved_sink = get_saved_config()
    result["saved_config_exists"] = saved_source is not None and saved_sink is not None
    
    if not result["saved_config_exists"]:
        result["issues"].append("No saved audio configuration. Run ./set_audio_defaults.sh")
    
    # Check current defaults
    current_source, current_sink = get_current_defaults()
    result["source_name"] = current_source
    result["sink_name"] = current_sink
    
    if saved_source:
        result["source_configured"] = current_source == saved_source
        if not result["source_configured"]:
            result["issues"].append(
                f"Microphone mismatch: expected '{saved_source}', got '{current_source}'"
            )
    
    if saved_sink:
        result["sink_configured"] = current_sink == saved_sink
        if not result["sink_configured"]:
            result["issues"].append(
                f"Speaker mismatch: expected '{saved_sink}', got '{current_sink}'"
            )
    
    result["all_ok"] = (
        result["pulseaudio_running"] and
        result["saved_config_exists"] and
        result["source_configured"] and
        result["sink_configured"]
    )
    
    return result


# Run validation when module is imported (with minimal output)
def _init_check():
    """Silent initialization check."""
    import sys
    # Only print warnings if not in test mode
    if "pytest" not in sys.modules:
        ensure_correct_audio_devices(verbose=False)


# Uncomment to auto-check on import:
# _init_check()


__all__ = [
    "AudioDeviceInfo",
    "get_saved_config",
    "get_current_defaults",
    "device_exists",
    "set_default_device",
    "ensure_correct_audio_devices",
    "get_pyaudio_device_index",
    "validate_audio_setup",
]
