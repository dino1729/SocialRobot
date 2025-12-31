"""
Unified Voice Assistant Entrypoint
===================================

This is the single entrypoint for the SocialRobot voice assistant. It consolidates the
previous `main_*.py` variants into one CLI, where features are enabled via flags.

High-level pipeline:
1) Listen: either continuous (VAD / fixed-time) or wake word → command capture
2) Transcribe: STT converts audio → text
3) Think: LLM generates a response (optionally with tool calls)
4) Speak: TTS synthesizes audio and plays it back

Configuration precedence:
- CLI flags override `.env`
- `.env` overrides built-in defaults in this file

FEATURE FLAGS:
  --wakeword          Enable wake word detection (default: disabled, continuous listening)
  --llm {ollama,litellm}  LLM backend selection (default: ollama for local inference)
  --tools             Enable internet tools: web search, URL scraping, weather (default: disabled)

EQUIVALENT COMMANDS to old scripts:
  main.py                                  -> python main.py
  main_wakeword.py                         -> python main.py --wakeword
  main_wakeword_online.py                  -> python main.py --wakeword --llm litellm
  main_internetconnected.py                -> python main.py --tools
  main_wakeword_internetconnected.py       -> python main.py --wakeword --tools
  main_wakeword_internetconnected_online.py -> python main.py --wakeword --llm litellm --tools

ADDITIONAL OPTIONS:
  --stt-engine {faster-whisper,openai-whisper}  STT engine (default: faster-whisper)
  --device {cpu,cuda}     Device for STT (auto-detected if not specified)
  --compute-type          Compute type for faster-whisper: int8, float16, float32 (auto-detected)
  --tts-engine {kokoro,piper,chatterbox,vibevoice}  TTS engine to use (default: kokoro)
  --tts-gpu               Enable GPU acceleration for TTS
  --tts-voice VOICE       Voice for Kokoro TTS (e.g., af_bella, af_sarah)
  --tts-speed SPEED       Speech speed for Kokoro TTS (default: 1.0)
  --voice NAME            Voice character name for Chatterbox (e.g., rick_sanchez, morgan_freeman)
                          Auto-loads voice from voices/{name}.wav and persona from personas/{name}.txt
  --vibevoice-speaker NAME  Speaker for VibeVoice TTS (default: Carter)
                          Available: Carter, Bria, Alex, Dora, Nova, Sol, Aria, Isla, Eva, Maya, Raj
  --wakeword-threshold    Wake word detection threshold 0.0-1.0 (default comes from WAKEWORD_THRESHOLD)
  --no-memory-monitor     Disable periodic memory usage stats
  --monitor-interval SEC  Memory monitor update interval in seconds (default: 60)
  --no-auto-calibrate     Disable automatic noise calibration at startup
  --calibration-seconds SEC  Duration of ambient noise sampling (default: 2.0)
  --debug                 Enable detailed debug logging for troubleshooting

NOISE CALIBRATION:
  At startup, the assistant measures ambient noise for 2 seconds (configurable) and
  automatically adjusts VAD aggressiveness and wakeword threshold for optimal performance:
  - Quiet rooms: More sensitive settings to catch soft speech
  - Noisy environments: Stricter filtering to reduce false triggers

EXAMPLES:
  # Basic voice assistant (continuous listening, local Ollama)
  python main.py

  # Wake word activated with local LLM
  python main.py --wakeword

  # Wake word with online LLM (requires LITELLM_API_KEY in .env)
  python main.py --wakeword --llm litellm

  # Internet-connected with web search and weather tools
  python main.py --tools

  # Full featured: wake word + online LLM + internet tools
  python main.py --wakeword --llm litellm --tools

  # Custom TTS settings
  python main.py --tts-engine piper --tts-gpu

  # Chatterbox TTS with Rick Sanchez voice and persona
  python main.py --tts-engine chatterbox --voice rick_sanchez --tts-gpu

  # Chatterbox TTS with Morgan Freeman voice
  python main.py --tts-engine chatterbox --voice morgan_freeman

  # VibeVoice TTS with GPU acceleration (recommended for speed)
  python main.py --tts-engine vibevoice --tts-gpu

  # VibeVoice with different speaker
  python main.py --tts-engine vibevoice --vibevoice-speaker Bria --tts-gpu

ENVIRONMENT VARIABLES (from .env file):
  # Feature Flags
  USE_WAKEWORD        - Enable wake word mode (default: false)
  LLM_BACKEND         - LLM backend: ollama or litellm (default: ollama)
  USE_TOOLS           - Enable internet tools (default: false)
  ENABLE_MEMORY_MONITOR - Show memory stats (default: true)
  MONITOR_INTERVAL    - Memory monitor interval in seconds (default: 60)
  AUTO_CALIBRATE      - Enable noise calibration (default: true)
  CALIBRATION_SECONDS - Calibration duration in seconds (default: 2.0)

  # LLM Configuration
  OLLAMA_URL          - Ollama API endpoint (default: http://localhost:11434/api/chat)
  OLLAMA_MODEL        - Ollama model to use (default: gemma3:270m or llama3.2:1b for tools)
  LITELLM_URL         - LiteLLM API endpoint (default: https://api.openai.com/v1/chat/completions)
  LITELLM_MODEL       - LiteLLM model (default: gpt-3.5-turbo)
  LITELLM_API_KEY     - API key for LiteLLM (required when using --llm litellm)

  # Wake Word
  WAKEWORD_MODEL      - Wake word model name (default: hey_jarvis_v0.1)
  WAKEWORD_THRESHOLD  - Detection threshold (default: 0.8 in code, often tuned per room)

  # VAD (Voice Activity Detection)
  USE_VAD             - Enable VAD for speech detection (default: true)
  FIXED_LISTEN_SECONDS - Recording duration when VAD disabled (default: 5.0)
  VAD_SAMPLE_RATE     - Audio sample rate in Hz (default: 16000)
  VAD_FRAME_DURATION_MS - Frame duration in ms: 10, 20, or 30 (default: 30)
  VAD_PADDING_DURATION_MS - Trailing audio duration in ms (default: 360)
  VAD_AGGRESSIVENESS  - Noise filtering 0-3, higher=stricter (default: 2)
  VAD_ACTIVATION_RATIO - Speech start threshold 0.0-1.0 (default: 0.6)
  VAD_DEACTIVATION_RATIO - Speech end threshold 0.0-1.0 (default: 0.85)

  # Conversation History Management
  MAX_CONVERSATION_MESSAGES - Max messages to keep in history (default: 20, 0=unlimited)
  MAX_CONVERSATION_TOKENS   - Max estimated tokens before auto-reset (default: 3500, 0=disabled)
  CONVERSATION_TIMEOUT_MINUTES - Minutes of inactivity before auto-reset (default: 10, 0=disabled)

  # Internet Tools
  FIRECRAWL_URL       - Firecrawl server URL for web tools (default: http://localhost:3002)
  OPENWEATHERMAP_API_KEY - API key for weather tool (optional)

  # STT Configuration
  STT_ENGINE          - STT engine (default: faster-whisper, options: openai-whisper)
  STT_GPU             - Enable STT GPU acceleration (default: false)
  STT_DEVICE          - STT device override (options: cpu, cuda)

  # TTS Configuration
  TTS_ENGINE          - TTS engine (default: kokoro, options: piper, chatterbox, vibevoice)
  TTS_GPU             - Enable TTS GPU acceleration (default: false)
  TTS_VOICE           - Voice character for Chatterbox (e.g., morgan_freeman, rick_sanchez)
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional, Callable, TYPE_CHECKING

# =============================================================================
# Suppress ALSA/JACK Warnings (before importing audio libraries)
# =============================================================================

# Best-effort suppression of noisy C-library logs (ALSA/JACK) so the terminal stays readable.
# This does not affect Python exceptions; it only silences some lower-level audio diagnostics.
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                          ctypes.c_char_p, ctypes.c_int,
                                          ctypes.c_char_p)
    def _py_error_handler(filename, line, function, err, fmt):
        pass
    _c_error_handler = ERROR_HANDLER_FUNC(_py_error_handler)
    _asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    _asound.snd_lib_error_set_handler(_c_error_handler)
except:
    pass

# =============================================================================
# Suppress Python Warnings (ONNX Runtime, webrtcvad, etc.)
# =============================================================================

import warnings

# Suppress ONNX Runtime warning about missing CUDA provider (harmless on CPU-only systems)
warnings.filterwarnings("ignore", message=".*Specified provider.*is not in available provider names.*")

# Suppress pkg_resources deprecation warning from webrtcvad
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Disable tqdm progress bars globally (faster-whisper uses them during transcription)
# This can be overridden by setting TQDM_DISABLE=0 in the environment
os.environ.setdefault("TQDM_DISABLE", "1")


@contextmanager
def suppress_stderr():
    """Temporarily redirect `stderr` to `/dev/null` (used around imports that spam warnings)."""
    sys.stderr.flush()
    old_stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)


# Import pyaudio with suppressed warnings
with suppress_stderr():
    import pyaudio  # noqa: F401 - imported for side effects (suppresses warnings on first use)

import requests
from dotenv import load_dotenv

# Core audio components (always needed)
from audio.vad import VADConfig, VADListener, FixedTimeListener
from audio.engine_config import create_tts_engine, create_stt_engine, TTSEngine, STTEngine

# Load configuration defaults from `.env` (CLI args may override these).
load_dotenv()

# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging with optional debug level.
    
    Args:
        debug: If True, enable DEBUG level logging
        
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Re-enable tqdm progress bars in debug mode
    if debug:
        os.environ["TQDM_DISABLE"] = "0"
    
    # Create formatter with timestamps
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Get root logger for the app
    logger = logging.getLogger('socialrobot')
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger (will be reconfigured in main() based on --debug flag)
logger = logging.getLogger('socialrobot')

# =============================================================================
# Configuration from Environment Variables
# =============================================================================

# These values are read once at import time and serve as defaults for CLI/config wiring.
# Keep names aligned with `env.example`.

# Ollama (local LLM) configuration.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")
# Tool-capable model: used when `--tools` is enabled because tool calling benefits from models
# trained for function/tool use.
OLLAMA_MODEL_TOOLS = os.getenv("OLLAMA_MODEL_TOOLS", "llama3.2:1b")

# LiteLLM (online LLM) configuration
LITELLM_URL = os.getenv("LITELLM_URL", "https://api.openai.com/v1/chat/completions")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gpt-3.5-turbo")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")

# Wake word configuration
WAKEWORD_MODEL = os.getenv("WAKEWORD_MODEL", "hey_jarvis_v0.1")
# Higher threshold reduces false positives; lower is more sensitive.
# Note: this is the in-code default if WAKEWORD_THRESHOLD is not set in `.env`.
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.8"))

# Internet tools configuration
FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")

# STT (Speech-to-Text) configuration
DEFAULT_STT_ENGINE = os.getenv("STT_ENGINE", "faster-whisper")
DEFAULT_STT_GPU = os.getenv("STT_GPU", "false").lower() in ("true", "1", "yes")
DEFAULT_STT_DEVICE = os.getenv("STT_DEVICE", None)  # None = auto-detect
# If STT_GPU is set but STT_DEVICE is not, derive device from STT_GPU
if DEFAULT_STT_DEVICE is None and DEFAULT_STT_GPU:
    DEFAULT_STT_DEVICE = "cuda"

# TTS (Text-to-Speech) configuration
DEFAULT_TTS_ENGINE = os.getenv("TTS_ENGINE", "kokoro")
DEFAULT_TTS_GPU = os.getenv("TTS_GPU", "false").lower() in ("true", "1", "yes")
DEFAULT_TTS_VOICE = os.getenv("TTS_VOICE", None)  # For Chatterbox voice character

# Feature flags configuration
DEFAULT_USE_WAKEWORD = os.getenv("USE_WAKEWORD", "false").lower() in ("true", "1", "yes")
DEFAULT_LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
DEFAULT_USE_TOOLS = os.getenv("USE_TOOLS", "false").lower() in ("true", "1", "yes")
DEFAULT_ENABLE_MEMORY_MONITOR = os.getenv("ENABLE_MEMORY_MONITOR", "true").lower() in ("true", "1", "yes")
DEFAULT_MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", "60"))
DEFAULT_AUTO_CALIBRATE = os.getenv("AUTO_CALIBRATE", "true").lower() in ("true", "1", "yes")
DEFAULT_CALIBRATION_SECONDS = float(os.getenv("CALIBRATION_SECONDS", "2.0"))
DEFAULT_FOLLOWUP_WINDOW_SECONDS = float(os.getenv("FOLLOWUP_WINDOW_SECONDS", "4.0"))

# VAD (Voice Activity Detection) configuration
DEFAULT_USE_VAD = os.getenv("USE_VAD", "true").lower() in ("true", "1", "yes")
DEFAULT_FIXED_LISTEN_SECONDS = float(os.getenv("FIXED_LISTEN_SECONDS", "5.0"))
DEFAULT_VAD_SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
DEFAULT_VAD_FRAME_DURATION_MS = int(os.getenv("VAD_FRAME_DURATION_MS", "30"))
DEFAULT_VAD_PADDING_DURATION_MS = int(os.getenv("VAD_PADDING_DURATION_MS", "360"))
DEFAULT_VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
DEFAULT_VAD_ACTIVATION_RATIO = float(os.getenv("VAD_ACTIVATION_RATIO", "0.6"))
DEFAULT_VAD_DEACTIVATION_RATIO = float(os.getenv("VAD_DEACTIVATION_RATIO", "0.85"))

# Conversation history management
DEFAULT_MAX_CONVERSATION_MESSAGES = int(os.getenv("MAX_CONVERSATION_MESSAGES", "20"))
DEFAULT_MAX_CONVERSATION_TOKENS = int(os.getenv("MAX_CONVERSATION_TOKENS", "3500"))
DEFAULT_CONVERSATION_TIMEOUT_MINUTES = int(os.getenv("CONVERSATION_TIMEOUT_MINUTES", "10"))


# =============================================================================
# Ollama Model Validation
# =============================================================================

def _get_ollama_base_url() -> str:
    """Extract the base Ollama URL from the chat endpoint."""
    # OLLAMA_URL is like http://localhost:11434/api/chat
    # We need http://localhost:11434 for the tags endpoint
    url = OLLAMA_URL.replace("/api/chat", "").rstrip("/")
    return url


def _check_ollama_available() -> bool:
    """Check if Ollama service is running."""
    try:
        base_url = _get_ollama_base_url()
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def _get_installed_ollama_models() -> list[str]:
    """Get list of installed Ollama models."""
    try:
        base_url = _get_ollama_base_url()
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except Exception:
        pass
    return []


def validate_ollama_model(model_name: str, use_tools: bool = False) -> bool:
    """Validate that the specified Ollama model is installed.
    
    Args:
        model_name: The model name to check (e.g., 'llama3.2:1b')
        use_tools: Whether tools mode is being used (for error message context)
    
    Returns:
        True if model is available, exits with error otherwise
    """
    # Check if Ollama is running
    if not _check_ollama_available():
        print("\n" + "=" * 70)
        print("ERROR: Ollama is not running!")
        print("=" * 70)
        print("\nPlease start Ollama with:")
        print("  ollama serve")
        print("\nOr check if it's running:")
        print("  curl http://localhost:11434/api/tags")
        print("=" * 70 + "\n")
        return False
    
    # Get installed models
    installed_models = _get_installed_ollama_models()
    
    if not installed_models:
        print("\n" + "=" * 70)
        print("ERROR: No Ollama models installed!")
        print("=" * 70)
        print("\nInstall a model with:")
        print("  ollama pull llama3.2:1b")
        print("=" * 70 + "\n")
        return False
    
    # Check if the requested model is installed
    if model_name not in installed_models:
        print("\n" + "=" * 70)
        print(f"ERROR: Model '{model_name}' is not installed!")
        print("=" * 70)
        print(f"\nThe model specified in your .env file is not available.")
        print(f"\nInstalled models:")
        for m in installed_models:
            print(f"  - {m}")
        print(f"\nTo fix this, either:")
        print(f"  1. Install the model:  ollama pull {model_name}")
        print(f"  2. Update .env to use an installed model:")
        if use_tools:
            print(f"     OLLAMA_MODEL_TOOLS={installed_models[0]}")
        else:
            print(f"     OLLAMA_MODEL={installed_models[0]}")
        print("=" * 70 + "\n")
        return False
    
    return True


# =============================================================================
# System Utility Functions
# =============================================================================

def _get_memory_stats() -> dict[str, float]:
    """Get system memory statistics from /proc/meminfo (Linux only, no external deps).
    
    Returns:
        Dictionary with keys: total, used, available, percent (all in MB except percent)
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        mem_total = mem_available = mem_free = 0
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1]) / 1024  # KB to MB
            elif line.startswith('MemAvailable:'):
                mem_available = int(line.split()[1]) / 1024
            elif line.startswith('MemFree:'):
                mem_free = int(line.split()[1]) / 1024
        
        mem_used = mem_total - mem_available
        mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
        
        return {
            'total': mem_total,
            'used': mem_used,
            'available': mem_available,
            'percent': mem_percent
        }
    except Exception:
        return {'total': 0, 'used': 0, 'available': 0, 'percent': 0}


def _get_process_memory() -> float:
    """Get current process RSS memory usage in MB from /proc/[pid]/status."""
    try:
        pid = os.getpid()
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # KB to MB
    except Exception:
        pass
    return 0.0


def _format_memory_stats(stats: dict[str, float], process_mem: float) -> str:
    """Format memory statistics for console display."""
    return (
        f"💾 RAM: {stats['used']:.0f}/{stats['total']:.0f}MB "
        f"({stats['percent']:.1f}%) | "
        f"Process: {process_mem:.0f}MB"
    )


def _memory_monitor(stop_event: threading.Event, interval: int = 60) -> None:
    """Background thread that periodically prints memory usage stats.
    
    Args:
        stop_event: Threading event to signal shutdown
        interval: Seconds between updates
    """
    while not stop_event.is_set():
        stats = _get_memory_stats()
        process_mem = _get_process_memory()
        print(f"\n{_format_memory_stats(stats, process_mem)}")
        stop_event.wait(interval)


def _detect_whisper_device() -> str:
    """Auto-detect best available device for faster-whisper (CUDA or CPU).
    
    Returns:
        'cuda' if CUDA available, otherwise 'cpu'
    """
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _is_jetson() -> bool:
    """Detect if running on NVIDIA Jetson platform (Orin Nano, etc.)."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            return "jetson" in model or "tegra" in model
    except Exception:
        pass
    return os.path.exists("/etc/nv_tegra_release")


def _get_default_compute_type(device: str) -> str:
    """Determine optimal compute type for faster-whisper based on hardware.
    
    Args:
        device: 'cuda' or 'cpu'
    
    Returns:
        'int8' for CPU/Jetson, 'float16' for desktop GPUs
    """
    if device == "cpu":
        return "int8"
    if _is_jetson():
        return "int8"  # Jetson Orin Nano works best with int8
    return "float16"  # Desktop GPUs (RTX series) use float16


# =============================================================================
# Pipeline Decision Helpers
# =============================================================================

# Minimum audio bytes for valid speech (1 second at 16kHz, 16-bit)
MIN_AUDIO_BYTES = 32000


def is_audio_too_short(raw_bytes: bytes, min_bytes: int = MIN_AUDIO_BYTES) -> bool:
    """Check if audio segment is too short to be valid speech.
    
    Args:
        raw_bytes: Raw audio bytes
        min_bytes: Minimum bytes threshold (default: 1 second at 16kHz)
        
    Returns:
        True if audio is too short and should be rejected
    """
    return len(raw_bytes) < min_bytes


def is_wakeword_echo(recognized_text: str, wakeword_phrases: list[str] = None) -> bool:
    """Check if recognized text is just the wake word (not a real command).
    
    Args:
        recognized_text: Text recognized from speech
        wakeword_phrases: List of wake word phrases to filter
        
    Returns:
        True if text is a wake word echo and should be ignored
    """
    if wakeword_phrases is None:
        wakeword_phrases = ["hey jarvis", "jarvis", "hey jarvis."]
    
    normalized = recognized_text.strip().lower()
    return normalized in wakeword_phrases


def is_self_echo(user_text: str, bot_response: str) -> bool:
    """Check if user speech is an echo of the bot's last response.
    
    This happens when the bot's TTS output is picked up by the microphone.
    
    Args:
        user_text: Text recognized from user speech
        bot_response: Last bot response text
        
    Returns:
        True if user text appears to be an echo of bot response
    """
    if not user_text or not bot_response:
        return False
    
    normalized_user = user_text.strip().lower()
    normalized_bot = bot_response.strip().lower()
    
    if not normalized_user or not normalized_bot:
        return False
    
    return (
        normalized_user == normalized_bot
        or normalized_user in normalized_bot
        or normalized_bot in normalized_user
    )


# =============================================================================
# Audio Calibration Helpers
# =============================================================================

def compute_noise_rms(audio_bytes: bytes) -> float:
    """Compute RMS (root mean square) of audio samples.
    
    Args:
        audio_bytes: Raw audio bytes in int16 format
        
    Returns:
        RMS value as float (0 for silence/empty input)
    """
    import numpy as np
    
    if not audio_bytes:
        return 0.0
    
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    
    # Compute RMS
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    return float(rms)


def compute_noise_dbfs(audio_bytes: bytes) -> float:
    """Compute dBFS (decibels relative to full scale) of audio samples.
    
    Args:
        audio_bytes: Raw audio bytes in int16 format
        
    Returns:
        dBFS value (0 = full scale, negative values = quieter)
    """
    import math
    
    rms = compute_noise_rms(audio_bytes)
    if rms <= 0:
        return -100.0  # Return very low value for silence
    
    # Full scale for int16 is 32767
    full_scale = 32767.0
    dbfs = 20 * math.log10(rms / full_scale)
    return dbfs


def compute_recommended_vad_aggressiveness(noise_dbfs: float) -> int:
    """Determine recommended VAD aggressiveness based on ambient noise level.
    
    Args:
        noise_dbfs: Ambient noise level in dBFS
        
    Returns:
        VAD aggressiveness level 0-3 (higher = more strict noise filtering)
    """
    # Thresholds determined empirically:
    # Very quiet (< -45 dBFS): Use low aggressiveness (more sensitive to speech)
    # Quiet (-45 to -35 dBFS): Use moderate-low aggressiveness
    # Moderate (-35 to -25 dBFS): Use moderate aggressiveness
    # Loud (> -25 dBFS): Use high aggressiveness (stricter filtering)
    
    if noise_dbfs < -45:
        return 1  # Quiet room - be more sensitive
    elif noise_dbfs < -35:
        return 2  # Normal room
    elif noise_dbfs < -25:
        return 2  # Moderate noise
    else:
        return 3  # Loud environment - be more strict


def compute_wakeword_threshold_boost(noise_dbfs: float) -> float:
    """Compute threshold boost for wakeword detection based on noise level.
    
    In noisier environments, we want to increase the threshold to reduce
    false positives. The boost is always >= 0 (never makes threshold more sensitive).
    
    Args:
        noise_dbfs: Ambient noise level in dBFS
        
    Returns:
        Threshold boost (0.0 to 0.3)
    """
    # Map noise level to boost:
    # Very quiet (< -45 dBFS): No boost needed
    # Quiet (-45 to -35 dBFS): Tiny boost (0.0-0.05)
    # Moderate (-35 to -25 dBFS): Small boost (0.05-0.15)
    # Loud (> -25 dBFS): Larger boost (0.15-0.3)
    
    if noise_dbfs < -45:
        return 0.0
    elif noise_dbfs < -35:
        # Linear interpolation from 0 to 0.05
        factor = (noise_dbfs + 45) / 10  # 0 to 1
        return 0.05 * factor
    elif noise_dbfs < -25:
        # Linear interpolation from 0.05 to 0.15
        factor = (noise_dbfs + 35) / 10  # 0 to 1
        return 0.05 + 0.10 * factor
    else:
        # Linear interpolation from 0.15 to 0.3, capped at 0.3
        factor = min((noise_dbfs + 25) / 10, 1.0)  # 0 to 1, capped
        return min(0.15 + 0.15 * factor, 0.3)


def calibrate_audio_parameters(
    sample_seconds: float = 2.0,
    base_wakeword_threshold: float = 0.5,
    sample_rate: int = 16000,
    device_index: Optional[int] = None,
) -> dict[str, Any]:
    """Measure ambient noise and compute recommended audio parameters.
    
    This function samples audio for a short period at startup to measure
    the background noise level, then computes recommended values for:
    - VAD aggressiveness (0-3)
    - Wakeword detection threshold
    
    Args:
        sample_seconds: How long to sample ambient noise (default 2 seconds)
        base_wakeword_threshold: Base wakeword threshold to boost from
        sample_rate: Audio sample rate in Hz
        device_index: PyAudio input device index (None = default)
        
    Returns:
        Dict with keys: vad_aggressiveness, wakeword_threshold, noise_dbfs, success
    """
    import pyaudio
    
    # Default values if calibration fails
    defaults = {
        'vad_aggressiveness': 3,  # Conservative default
        'wakeword_threshold': base_wakeword_threshold,
        'noise_dbfs': -40.0,
        'success': False,
    }
    
    try:
        # Suppress JACK "cannot connect to server" warnings during PyAudio init
        with suppress_stderr():
            pa = pyaudio.PyAudio()
        
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
            )
            
            # Collect audio samples
            total_samples = int(sample_rate * sample_seconds)
            chunk_size = 1024
            chunks = []
            
            samples_read = 0
            while samples_read < total_samples:
                chunk = stream.read(min(chunk_size, total_samples - samples_read), exception_on_overflow=False)
                chunks.append(chunk)
                samples_read += chunk_size
            
            stream.stop_stream()
            stream.close()
            
            # Combine all audio
            audio_bytes = b''.join(chunks)
            
            # Compute noise metrics
            noise_dbfs = compute_noise_dbfs(audio_bytes)
            
            # Compute recommended parameters
            vad_aggressiveness = compute_recommended_vad_aggressiveness(noise_dbfs)
            threshold_boost = compute_wakeword_threshold_boost(noise_dbfs)
            wakeword_threshold = min(base_wakeword_threshold + threshold_boost, 0.95)
            
            return {
                'vad_aggressiveness': vad_aggressiveness,
                'wakeword_threshold': wakeword_threshold,
                'noise_dbfs': noise_dbfs,
                'success': True,
            }
            
        finally:
            pa.terminate()
            
    except Exception as e:
        logger.warning(f"Audio calibration failed: {e}")
        return defaults


# =============================================================================
# Wake Word Detector (conditionally loaded when --wakeword is used)
# =============================================================================

class WakeWordDetector:
    """Wake word detector using OpenWakeWord library.
    
    This class is only instantiated when --wakeword flag is provided.
    It listens for a configured wake word (default: "hey jarvis") and triggers
    a callback when detected.
    """
    
    def __init__(
        self,
        wakeword_models: list[str],
        threshold: float = 0.5,
        chunk_size: int = 1280,
        sample_rate: int = 16000,
        device_index: Optional[int] = None,
    ) -> None:
        """Initialize wake word detector.
        
        Args:
            wakeword_models: List of wake word model names to load (e.g., ['hey_jarvis_v0.1'])
            threshold: Detection confidence threshold 0.0-1.0 (higher = stricter)
            chunk_size: Audio chunk size in samples
            sample_rate: Audio sample rate in Hz
            device_index: PyAudio input device index (None = default microphone)
        """
        import numpy as np
        import pyaudio
        from openwakeword.model import Model
        from pathlib import Path
        import openwakeword
        
        self.wakeword_models = wakeword_models
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.np = np  # Store numpy reference for use in other methods
        
        print(f"-> Loading wake word model(s): {', '.join(wakeword_models)}")
        # openwakeword 0.4.0 uses wakeword_model_paths for model file paths
        # Convert model names to full paths (e.g., 'hey_jarvis_v0.1' -> '/path/to/hey_jarvis_v0.1.onnx')
        oww_path = Path(openwakeword.__file__).parent
        models_dir = oww_path / "resources" / "models"
        model_paths = [str(models_dir / f"{model}.onnx") for model in wakeword_models]
        self.oww_model = Model(wakeword_model_paths=model_paths)
        
        # Suppress JACK "cannot connect to server" warnings during PyAudio init
        with suppress_stderr():
            self.audio = pyaudio.PyAudio()
        self.stream: Optional[Any] = None
        self.is_running = False
        self.is_paused = False
        self.detection_callback: Optional[Callable[[], None]] = None
        self.last_detection_time = 0.0
    
    def _play_bling_sound(self) -> None:
        """Play notification sound when wake word is detected (f1_beep.mp3)."""
        # Prevent double beeps with a lock
        if hasattr(self, '_beep_playing') and self._beep_playing:
            logger.debug("Beep already playing, skipping duplicate")
            return
        self._beep_playing = True
        logger.debug("Playing wake word beep sound")
        
        def play_sound():
            try:
                import subprocess
                script_dir = os.path.dirname(os.path.abspath(__file__))
                sound_path = os.path.join(script_dir, "f1_beep.mp3")
                
                if os.path.exists(sound_path):
                    logger.debug(f"Playing beep from: {sound_path}")
                    # Use subprocess.run to wait for completion, preventing overlaps
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", sound_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False
                    )
                    logger.debug("Beep playback completed")
                else:
                    logger.warning(f"Beep sound file not found: {sound_path}")
            except Exception as e:
                logger.error(f"Could not play beep sound: {e}")
                print(f"-> Warning: Could not play beep sound: {e}")
            finally:
                self._beep_playing = False
        
        # Run in background thread to avoid blocking
        sound_thread = threading.Thread(target=play_sound, daemon=True)
        sound_thread.start()
    
    def set_detection_callback(self, callback: Callable[[], None]) -> None:
        """Set callback function invoked when wake word is detected."""
        self.detection_callback = callback
    
    def pause(self) -> None:
        """Pause wake word detection (while processing a command)."""
        self.is_paused = True
        self.oww_model.reset()  # Clear model state to prevent false positives when resuming
        logger.debug("Wake word detector paused, model state cleared")
    
    def resume(self) -> None:
        """Resume wake word detection after command processing."""
        self.oww_model.reset()  # Clear internal buffer to prevent false positives
        self.last_detection_time = time.time()  # Reset debounce timer to prevent immediate triggers
        self.is_paused = False
        logger.debug("Wake word detector resumed with buffers cleared")
    
    def start(self) -> None:
        """Start listening for wake word in a blocking loop."""
        import pyaudio
        
        if self.is_running:
            return
        
        self.is_running = True
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )
        
        print(f"-> Wake word detector ready! Say '{self.wakeword_models[0].replace('_', ' ')}' to activate...")
        
        while self.is_running:
            try:
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                if self.is_paused:
                    # Discard audio while paused to prevent buffer buildup
                    continue
                
                audio_array = self.np.frombuffer(audio_data, dtype=self.np.int16)
                predictions = self.oww_model.predict(audio_array)
                
                # Debounce: minimum 3 seconds between detections to prevent double triggers
                current_time = time.time()
                for model_name, score in predictions.items():
                    # Only log scores above 0.3 to reduce noise
                    if score >= 0.3:
                        logger.debug(f"Wake word prediction: {model_name}={score:.3f} (threshold={self.threshold})")
                    if score >= self.threshold:
                        time_since_last = current_time - self.last_detection_time
                        logger.debug(f"Score above threshold. Time since last detection: {time_since_last:.2f}s")
                        if time_since_last >= 3.0:
                            print(f"🎤 Wake word detected! (confidence: {score:.2f})")
                            logger.info(f"Wake word '{model_name}' triggered with confidence {score:.2f}")
                            self.last_detection_time = current_time
                            self._play_bling_sound()
                            if self.detection_callback:
                                self.detection_callback()
                        else:
                            logger.debug(f"Ignoring detection - debounce active ({time_since_last:.2f}s < 3.0s)")
                        break
                
            except Exception as e:
                if self.is_running:
                    print(f"Wake word detection error: {e}")
                break
    
    def stop(self) -> None:
        """Stop wake word detection and release audio resources."""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.audio.terminate()


# =============================================================================
# Conversation History Management
# =============================================================================

class ConversationManager:
    """Manages conversation history with automatic truncation and timeout reset.
    
    Features:
    - Keeps conversation history within a maximum message count
    - Estimates token usage and resets when approaching limits
    - Auto-resets conversation after a configurable timeout period
    - Preserves system prompt across resets
    
    Configuration (from environment variables):
    - MAX_CONVERSATION_MESSAGES: Max messages to keep (default: 20)
    - MAX_CONVERSATION_TOKENS: Max estimated tokens before reset (default: 3500)
    - CONVERSATION_TIMEOUT_MINUTES: Minutes of inactivity before reset (default: 10)
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_messages: int = DEFAULT_MAX_CONVERSATION_MESSAGES,
        max_tokens: int = DEFAULT_MAX_CONVERSATION_TOKENS,
        timeout_minutes: int = DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
    ) -> None:
        """Initialize conversation manager.
        
        Args:
            system_prompt: Optional system prompt to preserve across resets
            max_messages: Maximum messages to keep (0 = unlimited)
            max_tokens: Maximum estimated tokens before reset (0 = disabled)
            timeout_minutes: Minutes of inactivity before reset (0 = disabled)
        """
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.timeout_minutes = timeout_minutes
        self.conversation_history: list[dict] = []
        self.last_interaction_time: float = time.time()
        
        # Initialize with system prompt if provided
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string.
        
        Uses a simple heuristic: ~4 characters per token on average.
        This is a rough approximation that works reasonably well for English text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        return len(text) // 4 + 1
    
    def _get_total_tokens(self) -> int:
        """Calculate total estimated tokens in conversation history.
        
        Returns:
            Total estimated tokens across all messages
        """
        total = 0
        for msg in self.conversation_history:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens(content)
        return total
    
    def _should_reset_for_timeout(self) -> bool:
        """Check if conversation should be reset due to inactivity timeout.
        
        Returns:
            True if timeout has elapsed since last interaction
        """
        if self.timeout_minutes <= 0:
            return False
        
        elapsed_minutes = (time.time() - self.last_interaction_time) / 60
        return elapsed_minutes >= self.timeout_minutes
    
    def _should_reset_for_tokens(self) -> bool:
        """Check if conversation should be reset due to token limit.
        
        Returns:
            True if estimated tokens exceed the configured maximum
        """
        if self.max_tokens <= 0:
            return False
        
        return self._get_total_tokens() >= self.max_tokens
    
    def _get_non_system_messages(self) -> list[dict]:
        """Get all messages except the system prompt.
        
        Returns:
            List of non-system messages
        """
        return [msg for msg in self.conversation_history if msg.get("role") != "system"]
    
    def _truncate_history(self) -> None:
        """Truncate conversation history to max_messages limit.
        
        Preserves the system prompt (if any) and keeps the most recent messages.
        Always keeps messages in pairs (user + assistant) when possible.
        """
        if self.max_messages <= 0:
            return
        
        non_system = self._get_non_system_messages()
        
        if len(non_system) <= self.max_messages:
            return
        
        # Keep only the most recent messages
        messages_to_keep = non_system[-self.max_messages:]
        
        # Rebuild history with system prompt first
        self.conversation_history = []
        if self.system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": self.system_prompt
            })
        self.conversation_history.extend(messages_to_keep)
        
        logger.debug(f"Truncated conversation history to {len(messages_to_keep)} messages")
    
    def reset_conversation(self, reason: str = "manual") -> None:
        """Reset conversation history, preserving only the system prompt.
        
        Args:
            reason: Reason for reset (for logging)
        """
        old_count = len(self._get_non_system_messages())
        self.conversation_history = []
        
        if self.system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        self.last_interaction_time = time.time()
        logger.info(f"Conversation reset ({reason}). Cleared {old_count} messages.")
    
    def check_and_manage_history(self) -> None:
        """Check limits and manage conversation history before each query.
        
        This should be called before adding a new user message.
        It will:
        1. Reset if timeout has elapsed
        2. Reset if token limit exceeded
        3. Truncate if message count exceeded
        """
        # Check timeout first
        if self._should_reset_for_timeout():
            elapsed = (time.time() - self.last_interaction_time) / 60
            self.reset_conversation(f"timeout after {elapsed:.1f} minutes")
            return
        
        # Check token limit
        if self._should_reset_for_tokens():
            tokens = self._get_total_tokens()
            self.reset_conversation(f"token limit reached (~{tokens} tokens)")
            return
        
        # Truncate if needed
        self._truncate_history()
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to conversation history.
        
        Args:
            content: User message content
        """
        self.check_and_manage_history()
        self.conversation_history.append({
            "role": "user",
            "content": content
        })
        self.last_interaction_time = time.time()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to conversation history.
        
        Args:
            content: Assistant message content
        """
        self.conversation_history.append({
            "role": "assistant",
            "content": content
        })
        self.last_interaction_time = time.time()
    
    def add_tool_message(self, content: str, tool_call_id: Optional[str] = None) -> None:
        """Add a tool result message to conversation history.
        
        Args:
            content: Tool result content
            tool_call_id: Optional tool call ID (for OpenAI format)
        """
        msg = {
            "role": "tool",
            "content": content
        }
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self.conversation_history.append(msg)
    
    def add_raw_message(self, message: dict) -> None:
        """Add a raw message dict to conversation history.
        
        Used for tool call messages that have special structure.
        
        Args:
            message: Message dict to add
        """
        self.conversation_history.append(message)
        self.last_interaction_time = time.time()
    
    def get_messages(self) -> list[dict]:
        """Get the current conversation history.
        
        Returns:
            List of message dicts
        """
        return self.conversation_history


# =============================================================================
# LLM Clients
# =============================================================================

class OllamaClient:
    """Simple Ollama client for basic chat (no tool support).
    
    Includes automatic conversation history management with:
    - Message count limiting (MAX_CONVERSATION_MESSAGES)
    - Token-based reset (MAX_CONVERSATION_TOKENS)
    - Timeout-based reset (CONVERSATION_TIMEOUT_MINUTES)
    """
    
    def __init__(
        self,
        url: str = "http://localhost:11434/api/chat",
        model: str = "gemma3:270m",
        stream: bool = True,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.stream = stream
        self._conversation = ConversationManager(system_prompt=system_prompt)
    
    @property
    def conversation_history(self) -> list[dict]:
        """Get conversation history (for compatibility)."""
        return self._conversation.get_messages()
    
    def query(self, user_text: str) -> str:
        """Send a query to Ollama and return the response."""
        self._conversation.add_user_message(user_text)
        
        payload = {
            "model": self.model,
            "messages": self._conversation.get_messages(),
            "stream": self.stream,
        }
        
        try:
            if self.stream:
                # Streaming response
                response = requests.post(self.url, json=payload, stream=True, timeout=120)
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "message" in data:
                            chunk = data["message"].get("content", "")
                            full_response += chunk
                
                self._conversation.add_assistant_message(full_response)
                return full_response.strip()
            else:
                response = requests.post(self.url, json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("message", {}).get("content", "")
                    self._conversation.add_assistant_message(content)
                    return content.strip()
                else:
                    return "Sorry, I encountered an error."
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Sorry, I encountered an error."


class OllamaClientWithTools:
    """Enhanced Ollama client that supports tool calling for internet features.
    
    Includes automatic conversation history management with:
    - Message count limiting (MAX_CONVERSATION_MESSAGES)
    - Token-based reset (MAX_CONVERSATION_TOKENS)
    - Timeout-based reset (CONVERSATION_TIMEOUT_MINUTES)
    """
    
    def __init__(
        self,
        url: str = "http://localhost:11434/api/chat",
        model: str = "llama3.2:1b",
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.stream = stream
        self._conversation = ConversationManager(system_prompt=system_prompt)
    
    @property
    def conversation_history(self) -> list[dict]:
        """Get conversation history (for compatibility)."""
        return self._conversation.get_messages()
    
    def query(self, user_text: str) -> str:
        """Query without tools (for compatibility)."""
        return self.query_with_tools(user_text, [])
    
    def query_with_tools(self, user_text: str, tools: list[dict]) -> str:
        """Query Ollama with tool support, executing tools as needed."""
        from tools import execute_tool_call
        
        self._conversation.add_user_message(user_text)
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            payload = {
                "model": self.model,
                "messages": self._conversation.get_messages(),
                "stream": self.stream,
                "tools": tools if tools else None
            }
            
            try:
                response = requests.post(self.url, json=payload, timeout=120)
                
                if response.status_code != 200:
                    print(f"Ollama error: {response.status_code}")
                    return "Sorry, I encountered an error processing your request."
                
                data = response.json()
                message = data.get("message", {})
                tool_calls = message.get("tool_calls", [])
                
                if tool_calls:
                    print(f"🔧 Model requested {len(tool_calls)} tool call(s)")
                    self._conversation.add_raw_message(message)
                    
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        tool_name = function.get("name", "")
                        arguments = function.get("arguments", {})
                        
                        print(f"   Calling: {tool_name}({arguments})")
                        tool_result = execute_tool_call(tool_name, arguments)
                        
                        self._conversation.add_tool_message(tool_result)
                    continue
                else:
                    content = message.get("content", "")
                    self._conversation.add_assistant_message(content)
                    return content.strip()
                    
            except Exception as e:
                print(f"Error querying Ollama: {e}")
                return "Sorry, I encountered an error processing your request."
        
        return "I apologize, but I couldn't complete your request after multiple attempts."


class LiteLLMClient:
    """LiteLLM client for online LLM APIs (OpenAI, Anthropic, etc.).
    
    Includes automatic conversation history management with:
    - Message count limiting (MAX_CONVERSATION_MESSAGES)
    - Token-based reset (MAX_CONVERSATION_TOKENS)
    - Timeout-based reset (CONVERSATION_TIMEOUT_MINUTES)
    """
    
    def __init__(
        self,
        url: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-3.5-turbo",
        api_key: str = "",
        stream: bool = True,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.stream = stream
        self.max_tokens = max_tokens
        self._conversation = ConversationManager(system_prompt=system_prompt)
    
    @property
    def conversation_history(self) -> list[dict]:
        """Get conversation history (for compatibility)."""
        return self._conversation.get_messages()
    
    def query(self, user_text: str) -> str:
        """Send a query to the LLM API and return the response."""
        self._conversation.add_user_message(user_text)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": self._conversation.get_messages(),
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        
        try:
            if self.stream:
                response = requests.post(self.url, json=payload, headers=headers, stream=True, timeout=120)
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith("data: "):
                            data_str = line_text[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            import json
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            chunk = delta.get("content", "")
                            full_response += chunk
                
                self._conversation.add_assistant_message(full_response)
                return full_response.strip()
            else:
                response = requests.post(self.url, json=payload, headers=headers, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    self._conversation.add_assistant_message(content)
                    return content.strip()
                else:
                    return "Sorry, I encountered an error."
        except Exception as e:
            print(f"LiteLLM error: {e}")
            return "Sorry, I encountered an error."


class LiteLLMClientWithTools:
    """LiteLLM client with tool calling support for internet features.
    
    Includes automatic conversation history management with:
    - Message count limiting (MAX_CONVERSATION_MESSAGES)
    - Token-based reset (MAX_CONVERSATION_TOKENS)
    - Timeout-based reset (CONVERSATION_TIMEOUT_MINUTES)
    """
    
    def __init__(
        self,
        url: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-3.5-turbo",
        api_key: str = "",
        stream: bool = False,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.stream = stream
        self.max_tokens = max_tokens
        self._conversation = ConversationManager(system_prompt=system_prompt)
    
    @property
    def conversation_history(self) -> list[dict]:
        """Get conversation history (for compatibility)."""
        return self._conversation.get_messages()
    
    def query(self, user_text: str) -> str:
        """Query without tools (for compatibility)."""
        return self.query_with_tools(user_text, [])
    
    def query_with_tools(self, user_text: str, tools: list[dict]) -> str:
        """Query LLM with tool support, executing tools as needed."""
        from tools import execute_tool_call
        import json
        
        self._conversation.add_user_message(user_text)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Convert Ollama-style tools to OpenAI format
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": tool["function"]
                })
            
            payload = {
                "model": self.model,
                "messages": self._conversation.get_messages(),
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            if openai_tools:
                payload["tools"] = openai_tools
            
            try:
                response = requests.post(self.url, json=payload, headers=headers, timeout=120)
                
                if response.status_code != 200:
                    print(f"LiteLLM error: {response.status_code}")
                    return "Sorry, I encountered an error processing your request."
                
                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])
                
                if tool_calls:
                    print(f"🔧 Model requested {len(tool_calls)} tool call(s)")
                    self._conversation.add_raw_message(message)
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("function", {}).get("name", "")
                        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                        
                        print(f"   Calling: {tool_name}({arguments})")
                        tool_result = execute_tool_call(tool_name, arguments)
                        
                        self._conversation.add_tool_message(tool_result, tool_call.get("id", ""))
                    continue
                else:
                    content = message.get("content", "")
                    self._conversation.add_assistant_message(content)
                    return content.strip()
                    
            except Exception as e:
                print(f"Error querying LiteLLM: {e}")
                return "Sorry, I encountered an error processing your request."
        
        return "I apologize, but I couldn't complete your request after multiple attempts."


# =============================================================================
# LLM Client Factory
# =============================================================================

def create_llm_client(
    llm_backend: str,
    use_tools: bool,
    system_prompt: str,
) -> Any:
    """Factory function to create the appropriate LLM client based on configuration.
    
    Args:
        llm_backend: 'ollama' or 'litellm'
        use_tools: Whether tool calling is needed
        system_prompt: System prompt for the LLM
    
    Returns:
        Configured LLM client instance
    """
    if llm_backend == "ollama":
        if use_tools:
            return OllamaClientWithTools(
                url=OLLAMA_URL,
                model=OLLAMA_MODEL_TOOLS,
                stream=False,  # Tools require non-streaming
                system_prompt=system_prompt,
            )
        else:
            return OllamaClient(
                url=OLLAMA_URL,
                model=OLLAMA_MODEL,
                stream=True,
                system_prompt=system_prompt,
            )
    else:  # litellm
        if use_tools:
            return LiteLLMClientWithTools(
                url=LITELLM_URL,
                model=LITELLM_MODEL,
                api_key=LITELLM_API_KEY,
                stream=False,  # Tools require non-streaming
                max_tokens=2048,
                system_prompt=system_prompt,
            )
        else:
            return LiteLLMClient(
                url=LITELLM_URL,
                model=LITELLM_MODEL,
                api_key=LITELLM_API_KEY,
                stream=True,
                max_tokens=2048,
                system_prompt=system_prompt,
            )


# =============================================================================
# System Prompt Generation
# =============================================================================

def get_system_prompt(use_tools: bool, persona_path: Optional[str] = None) -> str:
    """Generate the system prompt based on configuration.
    
    Args:
        use_tools: Whether internet tools are available
        persona_path: Optional path to a persona file to use instead of default prompt
    
    Returns:
        System prompt string
    """
    current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    
    # Load persona from file if provided
    if persona_path and os.path.exists(persona_path):
        try:
            with open(persona_path, 'r', encoding='utf-8') as f:
                base_prompt = f.read().strip()
            print(f"-> Loaded persona from: {persona_path}")
            
            # Add voice assistant context to persona
            base_prompt += "\n\nIMPORTANT: You are acting as a voice assistant. Keep your responses brief and conversational (1-2 sentences unless more detail is needed). Do not include emojis, asterisks, or formatting in your responses."
        except Exception as e:
            print(f"-> Warning: Failed to load persona file: {e}")
            base_prompt = None
    else:
        base_prompt = None
    
    # Fall back to default prompt if no persona loaded
    if base_prompt is None:
        base_prompt = """You are a world-class knowledgeable AI voice assistant, Orion, hosted on a Jetson Orin Nano Super. Your mission is to assist users with any questions or tasks they have on a wide range of topics. Use your knowledge, skills, and resources to provide accurate, relevant, and helpful responses. Please remember that you are a voice assistant and keep answers brief, concise and within 1-2 sentences, unless it's absolutely necessary to give a longer response. Be polite, friendly, and respectful in your interactions, and try to satisfy the user's needs as best as you can. Dont include any emojis or asterisks or any other formatting in your responses."""
    
    if use_tools:
        tools_prompt = f"""

IMPORTANT: Today's date is {current_datetime}.

You have access to tools, but ONLY use them when the user EXPLICITLY asks for:
- Weather information (use get_weather) - e.g., "What's the weather in Tokyo?"
- Current news or events (use search_web) - e.g., "What's the latest news about AI?"
- Specific web content (use scrape_url) - e.g., "Read the article at example.com"
- Stock prices, sports scores, or other real-time data (use search_web)

DO NOT use tools for:
- Greetings like "Hello", "How are you?", "Hi there"
- General knowledge questions you already know the answer to
- Math, definitions, explanations, jokes, or casual conversation
- Any question that doesn't require current/real-time information

When in doubt, answer directly WITHOUT using tools. Only use tools when the user clearly needs current information from the internet."""
        return base_prompt + tools_prompt
    
    return base_prompt


# =============================================================================
# Main Function
# =============================================================================

def main(
    use_wakeword: bool = False,
    llm_backend: str = "ollama",
    use_tools: bool = False,
    enable_memory_monitor: bool = True,
    monitor_interval: int = 60,
    stt_engine: str = "faster-whisper",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    wakeword_threshold: float = WAKEWORD_THRESHOLD,
    tts_engine: str = "kokoro",
    tts_use_gpu: bool = False,
    tts_voice: Optional[str] = None,
    tts_speed: float = 1.0,
    voice: Optional[str] = None,
    vibevoice_speaker: Optional[str] = None,
    auto_calibrate: bool = True,
    calibration_seconds: float = 2.0,
    use_vad: bool = True,
    fixed_listen_seconds: float = 5.0,
) -> None:
    """Main entry point for the voice assistant.

    Runtime overview:
    - Validates/repairs default audio devices (best-effort).
    - Optionally calibrates ambient noise to adjust VAD aggressiveness and wake word threshold.
    - Builds a listener (VAD or fixed-time) and wires callbacks:
      - Speech callback runs STT → LLM (optional tools) → TTS playback.
    - In wake word mode, the wake word detector runs continuously and starts the listener only
      after the wake word triggers.

    Args:
        use_wakeword: Enable wake word detection instead of continuous listening
        llm_backend: 'ollama' (local) or 'litellm' (online API)
        use_tools: Enable internet tools (web search, scrape, weather)
        enable_memory_monitor: Show periodic memory usage stats
        monitor_interval: Seconds between memory stat updates
        stt_engine: STT engine ('faster-whisper' or 'openai-whisper')
        device: Device for STT ('cpu' or 'cuda', auto-detected if None)
        compute_type: Compute type for STT ('int8', 'float16', 'float32', auto if None)
        wakeword_threshold: Wake word detection threshold 0.0-1.0
        tts_engine: TTS engine to use ('kokoro', 'piper', 'chatterbox', or 'vibevoice')
        tts_use_gpu: Enable GPU for TTS
        tts_voice: Voice for Kokoro TTS
        tts_speed: Speech speed for Kokoro TTS
        voice: Voice character name for Chatterbox (loads voice and persona automatically)
        vibevoice_speaker: Speaker name for VibeVoice TTS (default: Carter)
        auto_calibrate: Enable automatic noise calibration at startup
        calibration_seconds: Duration of noise sampling for calibration
        use_vad: Use VAD for speech detection (False = fixed-time recording)
        fixed_listen_seconds: Recording duration when VAD is disabled
    """
    # -------------------------------------------------------------------------
    # Audio Device Validation
    # -------------------------------------------------------------------------
    # Audio defaults drift on many Linux systems (reboots, USB hotplug, etc.). We validate that
    # the configured input/output devices exist and (if needed) try to select a reasonable USB
    # mic/speaker automatically. For interactive setup use: `./set_audio_defaults.sh`.
    from audio.audio_devices import ensure_correct_audio_devices, validate_audio_setup
    
    print("-> Checking audio configuration...")
    audio_status = validate_audio_setup()
    
    if not audio_status["all_ok"]:
        if audio_status["issues"]:
            for issue in audio_status["issues"]:
                print(f"   ⚠️  {issue}")
        
        # Attempt to fix automatically
        if ensure_correct_audio_devices(verbose=True):
            print("   ✓ Audio devices corrected automatically")
        else:
            print("   ⚠️  Could not configure audio devices automatically")
            print("   Run ./set_audio_defaults.sh to configure audio")
    else:
        print("   ✓ Audio devices configured correctly")
    
    # Log configuration at debug level
    logger.debug("=" * 60)
    logger.debug("Starting SocialRobot Voice Assistant")
    logger.debug("=" * 60)
    logger.debug(f"Configuration:")
    logger.debug(f"  Wake word: {use_wakeword} (threshold: {wakeword_threshold})")
    logger.debug(f"  LLM backend: {llm_backend}")
    logger.debug(f"  Tools enabled: {use_tools}")
    logger.debug(f"  STT engine: {stt_engine} (device: {device}, compute: {compute_type})")
    logger.debug(f"  TTS engine: {tts_engine} (GPU: {tts_use_gpu}, voice: {tts_voice or voice})")
    logger.debug(f"  Memory monitor: {enable_memory_monitor} (interval: {monitor_interval}s)")
    logger.debug("=" * 60)
    
    # Build mode description
    mode_parts = []
    if use_wakeword:
        mode_parts.append("wake word")
    else:
        mode_parts.append("continuous listening")
    mode_parts.append(f"{llm_backend} LLM")
    if use_tools:
        mode_parts.append("internet tools")
    mode_str = ", ".join(mode_parts)
    
    print(f"-> Initializing voice assistant ({mode_str})...")
    
    # Instantiate and validate the selected LLM backend.
    if llm_backend == "ollama":
        model_to_use = OLLAMA_MODEL_TOOLS if use_tools else OLLAMA_MODEL
        print(f"-> Ollama URL: {OLLAMA_URL}")
        print(f"-> Ollama Model: {model_to_use}")
        
        # Validate that the model is installed
        if not validate_ollama_model(model_to_use, use_tools):
            return  # Exit if model not available
    else:
        print(f"-> LiteLLM URL: {LITELLM_URL}")
        print(f"-> LiteLLM Model: {LITELLM_MODEL}")
        
        # Validate API key for LiteLLM
        if not LITELLM_API_KEY:
            print("-> ERROR: LITELLM_API_KEY is not set in .env file!")
            print("-> Please add LITELLM_API_KEY to your .env file and try again.")
            return
    
    if use_tools:
        print(f"-> Firecrawl URL: {FIRECRAWL_URL}")
        print(f"-> Weather API: {'✓ Configured' if OPENWEATHERMAP_API_KEY else '✗ Not configured'}")
        
        # Firecrawl is required for web search/scrape tools; warn early if unreachable.
        try:
            response = requests.get(f"{FIRECRAWL_URL}/", timeout=5)
            if response.status_code == 200:
                print("-> ✓ Firecrawl is accessible")
            else:
                print(f"-> ⚠️  Firecrawl returned status {response.status_code}")
        except Exception as e:
            print(f"-> ⚠️  Warning: Cannot reach Firecrawl at {FIRECRAWL_URL}: {e}")
    
    if use_wakeword:
        print(f"-> Wake Word: {WAKEWORD_MODEL}")
        print(f"-> Wake Word Threshold: {wakeword_threshold}")
    
    print(f"-> TTS Engine: {tts_engine.upper()}")
    print(f"-> TTS GPU: {'Enabled' if tts_use_gpu else 'Disabled'}")
    if voice:
        print(f"-> Voice Character: {voice}")
    
    # Display initial memory stats
    if enable_memory_monitor:
        stats = _get_memory_stats()
        process_mem = _get_process_memory()
        print(f"-> Initial {_format_memory_stats(stats, process_mem)}")
    
    # Start background memory monitor
    memory_stop_event = threading.Event()
    memory_thread = None
    if enable_memory_monitor:
        memory_thread = threading.Thread(
            target=_memory_monitor,
            args=(memory_stop_event, monitor_interval),
            daemon=True
        )
        memory_thread.start()
    
    # Automatic noise calibration:
    # Measures ambient noise briefly at startup and tunes parameters to reduce false triggers in
    # noisy environments while staying responsive in quiet rooms.
    calibrated_vad_aggressiveness = DEFAULT_VAD_AGGRESSIVENESS
    calibrated_wakeword_threshold = wakeword_threshold  # Use provided threshold as base
    
    if auto_calibrate:
        print(f"-> Calibrating audio parameters ({calibration_seconds:.1f}s)...")
        calibration_result = calibrate_audio_parameters(
            sample_seconds=calibration_seconds,
            base_wakeword_threshold=wakeword_threshold,
        )
        
        if calibration_result['success']:
            calibrated_vad_aggressiveness = calibration_result['vad_aggressiveness']
            calibrated_wakeword_threshold = calibration_result['wakeword_threshold']
            noise_dbfs = calibration_result['noise_dbfs']
            
            print(f"-> Calibration complete: noise={noise_dbfs:.1f} dBFS")
            print(f"   VAD aggressiveness: {calibrated_vad_aggressiveness} (0=sensitive, 3=strict)")
            if use_wakeword:
                print(f"   Wakeword threshold: {calibrated_wakeword_threshold:.2f} (base: {wakeword_threshold:.2f})")
        else:
            print(f"-> Calibration failed, using defaults")
    else:
        print(f"-> Auto-calibration disabled, using default parameters")
    
    # VAD configuration. Calibration only overrides aggressiveness; other values remain sourced
    # from `.env` defaults.
    vad_config = VADConfig(
        sample_rate=DEFAULT_VAD_SAMPLE_RATE,
        frame_duration_ms=DEFAULT_VAD_FRAME_DURATION_MS,
        padding_duration_ms=DEFAULT_VAD_PADDING_DURATION_MS,
        aggressiveness=calibrated_vad_aggressiveness,  # May be overridden by calibration
        activation_ratio=DEFAULT_VAD_ACTIVATION_RATIO,
        deactivation_ratio=DEFAULT_VAD_DEACTIVATION_RATIO,
    )
    
    # STT device selection: if the caller didn't specify `--device`, pick CUDA when available.
    stt_device = device if device else _detect_whisper_device()
    
    # Initialize STT engine. If GPU init fails (often cuDNN/CUDA issues), fall back to CPU.
    print(f"-> STT Engine: {stt_engine.upper()}")
    try:
        stt_model = create_stt_engine(
            engine=stt_engine,
            device=stt_device,
            model_size="tiny",
            compute_type=compute_type,
            language="en",
        )
    except Exception as e:
        # If CUDA fails (often due to missing cuDNN), try CPU fallback
        if "cuda" in str(e).lower() or "cudnn" in str(e).lower() or stt_device == "cuda":
            print(f"-> Warning: GPU STT failed ({e})")
            print(f"-> Falling back to CPU for STT...")
            stt_model = create_stt_engine(
                engine=stt_engine,
                device="cpu",
                model_size="tiny",
                compute_type="int8",
                language="en",
            )
        else:
            raise
    
    # Resolve voice + persona assets for Chatterbox.
    # `--voice <name>` maps to `voices/<name>.wav` + `personas/<name>.txt` when present.
    voice_path = None
    persona_path = None
    if voice:
        # Get the directory where main.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        voice_path = os.path.join(script_dir, "voices", f"{voice}.wav")
        persona_path = os.path.join(script_dir, "personas", f"{voice}.txt")
        
        # Validate voice file exists
        if not os.path.exists(voice_path):
            print(f"-> Warning: Voice file not found: {voice_path}")
            print(f"-> Available voices: dino, jensen_huang, morgan_freeman, morty_smith, rick_sanchez, sasha, satya_nadella, sundar_pichai")
            voice_path = None
        
        # Validate persona file exists
        if persona_path and not os.path.exists(persona_path):
            print(f"-> Warning: Persona file not found: {persona_path}")
            persona_path = None
    
    # Build the system prompt (optionally persona-based) and create an LLM client.
    system_prompt = get_system_prompt(use_tools, persona_path=persona_path)
    llm_client = create_llm_client(llm_backend, use_tools, system_prompt)
    
    # Load tool schemas (OpenAI-style function/tool definitions) when `--tools` is enabled.
    tools = []
    if use_tools:
        from tools import TOOLS
        tools = TOOLS
    
    # Initialize TTS engine. Engine-specific kwargs are routed through `audio/engine_config.py`.
    tts_kwargs = {"use_gpu": tts_use_gpu}
    if tts_engine == "kokoro":
        tts_kwargs["voice"] = tts_voice or "af_bella"
        tts_kwargs["speed"] = tts_speed
    elif tts_engine == "chatterbox":
        tts_kwargs["voice_path"] = voice_path
    elif tts_engine == "vibevoice":
        tts_kwargs["speaker"] = vibevoice_speaker or "Carter"
    tts_model = create_tts_engine(engine=tts_engine, **tts_kwargs)
    
    # Runtime state:
    # - `last_bot_response` supports simple echo suppression (mic picks up speaker output).
    # - In wake word mode, `listening_active` prevents re-entrancy while a command is being handled.
    vad_listener: Optional[VADListener] = None
    last_bot_response: str = ""
    listening_active = threading.Event() if use_wakeword else None
    wakeword_detector: Optional[WakeWordDetector] = None
    
    def on_speech_detected(raw_bytes: bytes) -> None:
        """Callback triggered when VAD detects speech has ended."""
        nonlocal vad_listener, last_bot_response, wakeword_detector
        
        logger.debug(f"VAD callback triggered with {len(raw_bytes)} bytes of audio")
        
        if vad_listener is None:
            logger.warning("VAD listener is None, ignoring callback")
            return
        
        # Filter out very short audio segments (likely noise)
        audio_duration_ms = (len(raw_bytes) / 32)  # 16kHz * 2 bytes = 32 bytes/ms
        logger.debug(f"Audio duration: {audio_duration_ms:.0f}ms ({len(raw_bytes)} bytes)")
        
        if is_audio_too_short(raw_bytes):
            logger.info(f"Ignoring short audio: {audio_duration_ms:.0f}ms < 1000ms minimum")
            print(f"-> Ignoring short audio segment ({len(raw_bytes)} bytes < {MIN_AUDIO_BYTES} min)")
            # Resume listening without processing
            if use_wakeword:
                listening_active.clear()
                time.sleep(0.3)
                wakeword_detector.resume()
                logger.debug("Resumed wake word detection after short audio filter")
            else:
                vad_listener.enable_vad()
                logger.debug("Re-enabled VAD after short audio filter")
            return
        
        vad_listener.disable_vad()
        logger.debug("VAD disabled for STT processing")
        
        logger.debug(f"Speech segment detected ({len(raw_bytes)} bytes)")
        logger.info(f"Processing speech segment: {audio_duration_ms:.0f}ms")
        
        # Perform STT
        stt_start = time.time()
        try:
            recognized_text = stt_model.run_stt(raw_bytes, sample_rate=vad_listener.sample_rate)
            stt_time = (time.time() - stt_start) * 1000
            logger.debug(f"STT completed in {stt_time:.0f}ms: '{recognized_text}'")
        except Exception as exc:
            logger.error(f"STT error: {exc}", exc_info=True)
            print("STT error:", exc)
            recognized_text = ""
        
        logger.info(f"User: {recognized_text}")
        
        # Helper to return to listening state. In wake word mode we stop the short-lived command
        # listener and resume wake word detection; in continuous mode we simply re-enable VAD.
        def resume_listening():
            if use_wakeword:
                vad_listener.stop()
                listening_active.clear()
                time.sleep(1.0)
                wakeword_detector.resume()
            else:
                vad_listener.enable_vad()
        
        if not recognized_text.strip():
            resume_listening()
            return
        
        # Filter wake word echoes (when using wake word mode)
        if use_wakeword and is_wakeword_echo(recognized_text):
            print("-> Ignoring wake word echo, waiting for actual command...")
            resume_listening()
            return
        
        # Echo cancellation: ignore if user speech matches last bot response
        if is_self_echo(recognized_text, last_bot_response):
            print("-> Ignoring self-echo from recent response.")
            resume_listening()
            return
        
        # Query LLM
        logger.debug(f"Sending query to LLM: '{recognized_text[:100]}...' (tools={use_tools})")
        llm_start = time.time()
        if use_tools:
            llm_response = llm_client.query_with_tools(recognized_text, tools)
        else:
            llm_response = llm_client.query(recognized_text)
        llm_time = (time.time() - llm_start) * 1000
        logger.debug(f"LLM response received in {llm_time:.0f}ms: '{llm_response[:100]}...'")
        
        logger.info(f"Bot: {llm_response}")
        
        if not llm_response.strip():
            logger.warning("LLM returned empty response")
            resume_listening()
            return
        
        # Synthesize and play TTS
        logger.debug(f"Starting TTS synthesis for {len(llm_response)} chars")
        tts_start = time.time()
        try:
            audio_data = tts_model.synthesize(llm_response)
            tts_time = (time.time() - tts_start) * 1000
            audio_duration = len(audio_data) / tts_model.sample_rate if hasattr(tts_model, 'sample_rate') else 0
            logger.debug(f"TTS synthesis completed in {tts_time:.0f}ms, audio duration: {audio_duration:.2f}s")
        except Exception as exc:
            logger.error(f"TTS error: {exc}", exc_info=True)
            print("TTS error:", exc)
            resume_listening()
            return
        
        last_bot_response = llm_response
        logger.debug("Starting audio playback")
        playback_start = time.time()
        tts_model.play_audio_with_amplitude(audio_data, amplitude_callback=None)
        playback_time = (time.time() - playback_start) * 1000
        logger.debug(f"Audio playback completed in {playback_time:.0f}ms")
        
        # Display memory stats after interaction
        if enable_memory_monitor:
            stats = _get_memory_stats()
            process_mem = _get_process_memory()
            print(f"-> {_format_memory_stats(stats, process_mem)}")
        
        # Return to listening state.
        # Wake word mode uses a longer cooldown to reduce immediate self-triggering from TTS audio.
        logger.debug("Returning to listening state")
        if use_wakeword:
            vad_listener.stop()
            listening_active.clear()
            
            # Brief follow-up window: listen for continuation without requiring wakeword
            followup_seconds = DEFAULT_FOLLOWUP_WINDOW_SECONDS
            if followup_seconds > 0:
                logger.debug(f"Starting {followup_seconds}s follow-up listening window")
                time.sleep(1.0)  # Let TTS audio clear first
                
                # Use a threading event to track if follow-up speech was detected
                followup_detected = threading.Event()
                
                def on_followup_speech(audio_data: bytes) -> None:
                    """Handle follow-up speech during the brief window."""
                    followup_detected.set()
                    # Process the follow-up speech recursively
                    on_speech_detected(audio_data)
                
                # Create a short fixed-time listener for follow-up
                followup_listener = FixedTimeListener(
                    listen_seconds=followup_seconds,
                    sample_rate=DEFAULT_VAD_SAMPLE_RATE,
                    device_index=None,
                    on_speech_callback=on_followup_speech
                )
                
                print("-> Continue speaking or say wake word for new topic...")
                followup_listener.start()
                
                # Wait for the follow-up window to complete
                # The listener will call on_followup_speech if speech is detected
                time.sleep(followup_seconds + 0.5)
                followup_listener.stop()
                
                # If follow-up was detected, the recursive call handles the rest
                if followup_detected.is_set():
                    logger.debug("Follow-up speech detected and processed")
                    return
                
                logger.debug("No follow-up speech detected, resuming wakeword")
            else:
                time.sleep(4.0)  # Original cooldown if follow-up disabled
            
            wakeword_detector.resume()
            logger.debug("Wake word detector resumed")
        else:
            vad_listener.enable_vad()
            logger.debug("VAD re-enabled")
    
    # Wake word callback: pauses the detector, then starts a listener to capture one command.
    def on_wakeword_detected() -> None:
        nonlocal vad_listener, wakeword_detector
        
        if listening_active.is_set():
            return
        
        listening_active.set()
        wakeword_detector.pause()
        
        print("-> Listening for command...")
        time.sleep(0.5)  # Let wake word audio clear
        
        # Create listener based on VAD setting
        if use_vad:
            vad_listener = VADListener(
                config=vad_config,
                device_index=None,
                on_speech_callback=on_speech_detected
            )
        else:
            vad_listener = FixedTimeListener(
                listen_seconds=fixed_listen_seconds,
                sample_rate=DEFAULT_VAD_SAMPLE_RATE,
                device_index=None,
                on_speech_callback=on_speech_detected
            )
        vad_listener.start()
    
    # Start the main loop based on mode
    if use_wakeword:
        wakeword_detector = WakeWordDetector(
            wakeword_models=[WAKEWORD_MODEL],
            threshold=calibrated_wakeword_threshold,
            chunk_size=1280,
            sample_rate=16000,
            device_index=None,
        )
        wakeword_detector.set_detection_callback(on_wakeword_detected)
        
        print("-> Voice assistant ready with wake word detection!")
    else:
        # Create listener based on VAD setting
        if use_vad:
            vad_listener = VADListener(
                config=vad_config,
                device_index=None,
                on_speech_callback=on_speech_detected
            )
            print("-> Voice assistant ready! Listening with VAD...")
        else:
            vad_listener = FixedTimeListener(
                listen_seconds=fixed_listen_seconds,
                sample_rate=DEFAULT_VAD_SAMPLE_RATE,
                device_index=None,
                on_speech_callback=on_speech_detected
            )
            print(f"-> Voice assistant ready! Fixed {fixed_listen_seconds}s recording mode...")
    
    if use_tools:
        print("-> 🌐 Internet access enabled via Firecrawl tools")
        if OPENWEATHERMAP_API_KEY:
            print("-> 🌤️  Weather information enabled via OpenWeatherMap")
    
    if enable_memory_monitor:
        print(f"-> Memory monitoring enabled (updates every {monitor_interval}s)")
    
    print("-> Press Ctrl+C to stop.")
    
    try:
        if use_wakeword:
            wakeword_detector.start()
        else:
            vad_listener.start()
    except KeyboardInterrupt:
        print("\n-> Shutting down...")
    finally:
        # Cleanup
        if use_wakeword and wakeword_detector:
            wakeword_detector.stop()
        if vad_listener:
            vad_listener.stop()
        
        if enable_memory_monitor:
            memory_stop_event.set()
            if memory_thread and memory_thread.is_alive():
                memory_thread.join(timeout=1)
        
        if enable_memory_monitor:
            stats = _get_memory_stats()
            process_mem = _get_process_memory()
            print(f"-> Final {_format_memory_stats(stats, process_mem)}")
        
        print("-> Goodbye!")


# =============================================================================
# CLI Argument Parser
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Voice Assistant with configurable features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mode Examples:
  # Basic voice assistant (continuous listening, local Ollama)
  python main.py
  
  # Wake word activated (say "hey jarvis" to activate)
  python main.py --wakeword
  
  # Wake word with online LLM (requires LITELLM_API_KEY in .env)
  python main.py --wakeword --llm litellm
  
  # Internet-connected with web search and weather tools
  python main.py --tools
  
  # Full featured: wake word + online LLM + internet tools
  python main.py --wakeword --llm litellm --tools
  
TTS/STT Examples:
  # Use Piper TTS with GPU
  python main.py --tts-engine piper --tts-gpu
  
  # Custom Kokoro voice and speed
  python main.py --tts-voice af_sarah --tts-speed 1.2
  
  # Chatterbox TTS with Rick Sanchez voice and persona
  python main.py --tts-engine chatterbox --voice rick_sanchez --tts-gpu
  
  # Chatterbox TTS with Morgan Freeman voice (narration style)
  python main.py --tts-engine chatterbox --voice morgan_freeman

  # VibeVoice TTS with GPU (recommended for ~300ms latency)
  python main.py --tts-engine vibevoice --tts-gpu

  # VibeVoice with different speaker
  python main.py --tts-engine vibevoice --vibevoice-speaker Bria --tts-gpu

  # Force CPU for STT (if GPU fails with cuDNN errors)
  python main.py --device cpu
  
  # Use OpenAI Whisper STT (better GPU compatibility, no cuDNN needed)
  python main.py --stt-engine openai-whisper
  
  # Force specific compute type for faster-whisper STT
  python main.py --compute-type int8

Equivalent to old scripts:
  main.py                                  -> python main.py
  main_wakeword.py                         -> python main.py --wakeword
  main_wakeword_online.py                  -> python main.py --wakeword --llm litellm
  main_internetconnected.py                -> python main.py --tools
  main_wakeword_internetconnected.py       -> python main.py --wakeword --tools
  main_wakeword_internetconnected_online.py -> python main.py --wakeword --llm litellm --tools
        """,
    )
    
    # Feature flags
    parser.add_argument(
        "--wakeword",
        action="store_true",
        default=DEFAULT_USE_WAKEWORD,
        help=f"Enable wake word detection (default: {DEFAULT_USE_WAKEWORD})",
    )
    
    parser.add_argument(
        "--llm",
        type=str,
        choices=["ollama", "litellm"],
        default=DEFAULT_LLM_BACKEND,
        help=f"LLM backend: 'ollama' for local, 'litellm' for online API (default: {DEFAULT_LLM_BACKEND})",
    )
    
    parser.add_argument(
        "--tools",
        action="store_true",
        default=DEFAULT_USE_TOOLS,
        help=f"Enable internet tools: web search, URL scraping, weather (default: {DEFAULT_USE_TOOLS})",
    )
    
    # STT options
    parser.add_argument(
        "--stt-engine",
        type=str,
        choices=["faster-whisper", "openai-whisper"],
        default=DEFAULT_STT_ENGINE,
        help=f"STT engine: 'faster-whisper' (ctranslate2, needs cuDNN for GPU) or "
             f"'openai-whisper' (PyTorch, better GPU compatibility). Default: {DEFAULT_STT_ENGINE}",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device for STT. Auto-detected if not specified. Use 'cpu' if GPU fails.",
    )
    
    parser.add_argument(
        "--compute-type",
        type=str,
        choices=["int8", "float16", "float32"],
        default=None,
        help="Compute type for faster-whisper. Auto-detected if not specified.",
    )
    
    # Wake word options
    parser.add_argument(
        "--wakeword-threshold",
        type=float,
        default=WAKEWORD_THRESHOLD,
        help=f"Wake word detection threshold 0.0-1.0 (default: {WAKEWORD_THRESHOLD})",
    )
    
    # TTS options
    parser.add_argument(
        "--tts-engine",
        type=str,
        choices=["kokoro", "piper", "chatterbox", "vibevoice"],
        default=DEFAULT_TTS_ENGINE,
        help=f"TTS engine to use (default: {DEFAULT_TTS_ENGINE})",
    )
    
    parser.add_argument(
        "--tts-gpu",
        action="store_true",
        default=DEFAULT_TTS_GPU,
        help=f"Enable GPU acceleration for TTS (default: {DEFAULT_TTS_GPU})",
    )
    
    parser.add_argument(
        "--tts-voice",
        type=str,
        default=None,
        help="Voice for Kokoro TTS (e.g., af_bella, af_sarah, am_adam, bf_emma)",
    )
    
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="Speech speed for Kokoro TTS (default: 1.0)",
    )
    
    parser.add_argument(
        "--voice",
        type=str,
        default=DEFAULT_TTS_VOICE,
        help="Voice character name for Chatterbox TTS (e.g., rick_sanchez, morgan_freeman). "
             "Auto-loads voice WAV from voices/ and persona from personas/",
    )

    parser.add_argument(
        "--vibevoice-speaker",
        type=str,
        default=None,
        help="Speaker for VibeVoice TTS (default: Carter). "
             "Available: Carter, Bria, Alex, Dora, Nova, Sol, Aria, Isla, Eva, Maya, Raj",
    )

    # Monitoring options
    parser.add_argument(
        "--no-memory-monitor",
        action="store_true",
        default=not DEFAULT_ENABLE_MEMORY_MONITOR,
        help=f"Disable memory usage monitoring (default monitoring: {DEFAULT_ENABLE_MEMORY_MONITOR})",
    )
    
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=DEFAULT_MONITOR_INTERVAL,
        help=f"Memory monitor update interval in seconds (default: {DEFAULT_MONITOR_INTERVAL})",
    )
    
    # Calibration options
    parser.add_argument(
        "--no-auto-calibrate",
        action="store_true",
        default=not DEFAULT_AUTO_CALIBRATE,
        help=f"Disable automatic noise calibration at startup (default calibration: {DEFAULT_AUTO_CALIBRATE})",
    )
    
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=DEFAULT_CALIBRATION_SECONDS,
        help=f"Duration of ambient noise sampling for calibration in seconds (default: {DEFAULT_CALIBRATION_SECONDS})",
    )
    
    # VAD options
    parser.add_argument(
        "--no-vad",
        action="store_true",
        default=not DEFAULT_USE_VAD,
        help=f"Disable VAD, use fixed-time recording instead (default VAD: {DEFAULT_USE_VAD})",
    )
    
    parser.add_argument(
        "--fixed-listen-seconds",
        type=float,
        default=DEFAULT_FIXED_LISTEN_SECONDS,
        help=f"Recording duration in seconds when VAD is disabled (default: {DEFAULT_FIXED_LISTEN_SECONDS})",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging for troubleshooting",
    )
    
    args = parser.parse_args()
    
    # Setup logging based on debug flag
    setup_logging(debug=args.debug)
    
    main(
        use_wakeword=args.wakeword,
        llm_backend=args.llm,
        use_tools=args.tools,
        enable_memory_monitor=not args.no_memory_monitor,
        monitor_interval=args.monitor_interval,
        stt_engine=args.stt_engine,
        device=args.device,
        compute_type=args.compute_type,
        wakeword_threshold=args.wakeword_threshold,
        tts_engine=args.tts_engine,
        tts_use_gpu=args.tts_gpu,
        tts_voice=args.tts_voice,
        tts_speed=args.tts_speed,
        voice=args.voice,
        vibevoice_speaker=args.vibevoice_speaker,
        auto_calibrate=not args.no_auto_calibrate,
        calibration_seconds=args.calibration_seconds,
        use_vad=not args.no_vad,
        fixed_listen_seconds=args.fixed_listen_seconds,
    )
