"""
Unified Voice Assistant Entrypoint
===================================

This consolidated script replaces the previous 6 separate main_*.py scripts with a single
unified CLI that supports all feature combinations through command-line flags.

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
  --tts-engine {kokoro,piper,chatterbox}  TTS engine to use (default: kokoro)
  --tts-gpu               Enable GPU acceleration for TTS
  --tts-voice VOICE       Voice for Kokoro TTS (e.g., af_bella, af_sarah)
  --tts-speed SPEED       Speech speed for Kokoro TTS (default: 1.0)
  --voice NAME            Voice character name for Chatterbox (e.g., rick_sanchez, morgan_freeman)
                          Auto-loads voice from voices/{name}.wav and persona from personas/{name}.txt
  --wakeword-threshold    Wake word detection threshold 0.0-1.0 (default: 0.5)
  --no-memory-monitor     Disable periodic memory usage stats
  --monitor-interval SEC  Memory monitor update interval in seconds (default: 60)
  --debug                 Enable detailed debug logging for troubleshooting

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

ENVIRONMENT VARIABLES (from .env file):
  OLLAMA_URL          - Ollama API endpoint (default: http://localhost:11434/api/chat)
  OLLAMA_MODEL        - Ollama model to use (default: gemma3:270m or llama3.2:1b for tools)
  LITELLM_URL         - LiteLLM API endpoint (default: https://api.openai.com/v1/chat/completions)
  LITELLM_MODEL       - LiteLLM model (default: gpt-3.5-turbo)
  LITELLM_API_KEY     - API key for LiteLLM (required when using --llm litellm)
  WAKEWORD_MODEL      - Wake word model name (default: hey_jarvis_v0.1)
  WAKEWORD_THRESHOLD  - Detection threshold (default: 0.5)
  FIRECRAWL_URL       - Firecrawl server URL for web tools (default: http://localhost:3002)
  OPENWEATHERMAP_API_KEY - API key for weather tool (optional)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Optional, Callable, TYPE_CHECKING

import requests
from dotenv import load_dotenv

# Core audio components (always needed)
from audio.vad import VADConfig, VADListener
from audio.engine_config import create_tts_engine, create_stt_engine, TTSEngine, STTEngine

# Load environment variables from .env file
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

# Ollama (local LLM) configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")
OLLAMA_MODEL_TOOLS = os.getenv("OLLAMA_MODEL_TOOLS", "llama3.2:1b")  # Tool-capable model

# LiteLLM (online LLM) configuration
LITELLM_URL = os.getenv("LITELLM_URL", "https://api.openai.com/v1/chat/completions")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gpt-3.5-turbo")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")

# Wake word configuration
WAKEWORD_MODEL = os.getenv("WAKEWORD_MODEL", "hey_jarvis_v0.1")
# Higher threshold (0.8) reduces false positives; lower (0.5) is more sensitive
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.8"))

# Internet tools configuration
FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")


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
        
        self.wakeword_models = wakeword_models
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.np = np  # Store numpy reference for use in other methods
        
        print(f"-> Loading wake word model(s): {', '.join(wakeword_models)}")
        # openwakeword uses wakeword_models for pre-trained model names (e.g., 'hey_jarvis_v0.1')
        # and wakeword_model_paths for custom model file paths
        self.oww_model = Model(wakeword_models=wakeword_models, inference_framework="onnx")
        
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
# LLM Clients
# =============================================================================

class OllamaClient:
    """Simple Ollama client for basic chat (no tool support)."""
    
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
        self.conversation_history: list[dict] = []
        
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def query(self, user_text: str) -> str:
        """Send a query to Ollama and return the response."""
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
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
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                return full_response.strip()
            else:
                response = requests.post(self.url, json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("message", {}).get("content", "")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": content
                    })
                    return content.strip()
                else:
                    return "Sorry, I encountered an error."
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Sorry, I encountered an error."


class OllamaClientWithTools:
    """Enhanced Ollama client that supports tool calling for internet features."""
    
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
        self.system_prompt = system_prompt
        self.conversation_history: list[dict] = []
        
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def query(self, user_text: str) -> str:
        """Query without tools (for compatibility)."""
        return self.query_with_tools(user_text, [])
    
    def query_with_tools(self, user_text: str, tools: list[dict]) -> str:
        """Query Ollama with tool support, executing tools as needed."""
        from tools import execute_tool_call
        
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            payload = {
                "model": self.model,
                "messages": self.conversation_history,
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
                    self.conversation_history.append(message)
                    
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        tool_name = function.get("name", "")
                        arguments = function.get("arguments", {})
                        
                        print(f"   Calling: {tool_name}({arguments})")
                        tool_result = execute_tool_call(tool_name, arguments)
                        
                        self.conversation_history.append({
                            "role": "tool",
                            "content": tool_result
                        })
                    continue
                else:
                    content = message.get("content", "")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": content
                    })
                    return content.strip()
                    
            except Exception as e:
                print(f"Error querying Ollama: {e}")
                return "Sorry, I encountered an error processing your request."
        
        return "I apologize, but I couldn't complete your request after multiple attempts."


class LiteLLMClient:
    """LiteLLM client for online LLM APIs (OpenAI, Anthropic, etc.)."""
    
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
        self.conversation_history: list[dict] = []
        
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def query(self, user_text: str) -> str:
        """Send a query to the LLM API and return the response."""
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
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
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                return full_response.strip()
            else:
                response = requests.post(self.url, json=payload, headers=headers, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": content
                    })
                    return content.strip()
                else:
                    return "Sorry, I encountered an error."
        except Exception as e:
            print(f"LiteLLM error: {e}")
            return "Sorry, I encountered an error."


class LiteLLMClientWithTools:
    """LiteLLM client with tool calling support for internet features."""
    
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
        self.conversation_history: list[dict] = []
        
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def query(self, user_text: str) -> str:
        """Query without tools (for compatibility)."""
        return self.query_with_tools(user_text, [])
    
    def query_with_tools(self, user_text: str, tools: list[dict]) -> str:
        """Query LLM with tool support, executing tools as needed."""
        from tools import execute_tool_call
        import json
        
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
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
                "messages": self.conversation_history,
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
                    self.conversation_history.append(message)
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("function", {}).get("name", "")
                        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                        
                        print(f"   Calling: {tool_name}({arguments})")
                        tool_result = execute_tool_call(tool_name, arguments)
                        
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": tool_result
                        })
                    continue
                else:
                    content = message.get("content", "")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": content
                    })
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
) -> None:
    """Main entry point for the voice assistant.
    
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
        tts_engine: TTS engine to use ('kokoro', 'piper', or 'chatterbox')
        tts_use_gpu: Enable GPU for TTS
        tts_voice: Voice for Kokoro TTS
        tts_speed: Speech speed for Kokoro TTS
        voice: Voice character name for Chatterbox (loads voice and persona automatically)
    """
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
    
    # Print configuration and validate
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
        
        # Check Firecrawl connectivity
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
    
    # VAD configuration
    # aggressiveness: 0=least aggressive (sensitive), 3=most aggressive (filters noise better)
    vad_config = VADConfig(
        sample_rate=16000,
        frame_duration_ms=30,
        padding_duration_ms=360,
        aggressiveness=3  # Maximum noise filtering
    )
    
    # Device detection for STT
    stt_device = device if device else _detect_whisper_device()
    
    # Initialize STT model using the factory
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
    
    # Resolve voice paths for Chatterbox TTS
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
    
    # Initialize LLM client with optional persona
    system_prompt = get_system_prompt(use_tools, persona_path=persona_path)
    llm_client = create_llm_client(llm_backend, use_tools, system_prompt)
    
    # Load tools if enabled
    tools = []
    if use_tools:
        from tools import TOOLS
        tools = TOOLS
    
    # Initialize TTS engine
    tts_kwargs = {"use_gpu": tts_use_gpu}
    if tts_engine == "kokoro":
        tts_kwargs["voice"] = tts_voice or "af_bella"
        tts_kwargs["speed"] = tts_speed
    elif tts_engine == "chatterbox":
        tts_kwargs["voice_path"] = voice_path
    tts_model = create_tts_engine(engine=tts_engine, **tts_kwargs)
    
    # State variables
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
        # At 16kHz, 16-bit audio: 32000 bytes = 1 second
        # Require at least 1 second of audio for valid speech
        MIN_AUDIO_BYTES = 32000  # ~1 second minimum
        audio_duration_ms = (len(raw_bytes) / 32) # 16kHz * 2 bytes = 32 bytes/ms
        logger.debug(f"Audio duration: {audio_duration_ms:.0f}ms ({len(raw_bytes)} bytes)")
        
        if len(raw_bytes) < MIN_AUDIO_BYTES:
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
        
        print(f"-> Speech segment detected ({len(raw_bytes)} bytes).")
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
        
        print("-> User said:", recognized_text)
        
        # Helper to return to listening state
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
        
        normalized_user = recognized_text.strip().lower()
        
        # Filter wake word echoes (when using wake word mode)
        if use_wakeword:
            wake_word_phrases = ["hey jarvis", "jarvis", "hey jarvis."]
            if normalized_user in wake_word_phrases:
                print("-> Ignoring wake word echo, waiting for actual command...")
                resume_listening()
                return
        
        # Echo cancellation: ignore if user speech matches last bot response
        normalized_bot = last_bot_response.strip().lower()
        if normalized_user and normalized_bot:
            if (
                normalized_user == normalized_bot
                or normalized_user in normalized_bot
                or normalized_bot in normalized_user
            ):
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
        
        print("-> Bot response:", llm_response)
        
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
        
        # Return to listening state
        logger.debug("Returning to listening state")
        if use_wakeword:
            vad_listener.stop()
            listening_active.clear()
            time.sleep(4.0)  # Longer cooldown to let TTS audio fully clear before resuming wake word
            wakeword_detector.resume()
            logger.debug("Wake word detector resumed after 4s cooldown")
        else:
            vad_listener.enable_vad()
            logger.debug("VAD re-enabled")
    
    # Wake word mode: define callback for when wake word is detected
    def on_wakeword_detected() -> None:
        nonlocal vad_listener, wakeword_detector
        
        if listening_active.is_set():
            return
        
        listening_active.set()
        wakeword_detector.pause()
        
        print("-> Listening for command...")
        time.sleep(0.5)  # Let wake word audio clear
        
        vad_listener = VADListener(
            config=vad_config,
            device_index=None,
            on_speech_callback=on_speech_detected
        )
        vad_listener.start()
    
    # Start the main loop based on mode
    if use_wakeword:
        wakeword_detector = WakeWordDetector(
            wakeword_models=[WAKEWORD_MODEL],
            threshold=wakeword_threshold,
            chunk_size=1280,
            sample_rate=16000,
            device_index=None,
        )
        wakeword_detector.set_detection_callback(on_wakeword_detected)
        
        print("-> Voice assistant ready with wake word detection!")
    else:
        vad_listener = VADListener(
            config=vad_config,
            device_index=None,
            on_speech_callback=on_speech_detected
        )
        print("-> Voice assistant ready! Listening for speech...")
    
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
        help="Enable wake word detection (default: continuous listening mode)",
    )
    
    parser.add_argument(
        "--llm",
        type=str,
        choices=["ollama", "litellm"],
        default="ollama",
        help="LLM backend: 'ollama' for local, 'litellm' for online API (default: ollama)",
    )
    
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Enable internet tools: web search, URL scraping, weather",
    )
    
    # STT options
    parser.add_argument(
        "--stt-engine",
        type=str,
        choices=["faster-whisper", "openai-whisper"],
        default="faster-whisper",
        help="STT engine: 'faster-whisper' (ctranslate2, needs cuDNN for GPU) or "
             "'openai-whisper' (PyTorch, better GPU compatibility). Default: faster-whisper",
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
        choices=["kokoro", "piper", "chatterbox"],
        default="kokoro",
        help="TTS engine to use (default: kokoro)",
    )
    
    parser.add_argument(
        "--tts-gpu",
        action="store_true",
        help="Enable GPU acceleration for TTS",
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
        default=None,
        help="Voice character name for Chatterbox TTS (e.g., rick_sanchez, morgan_freeman). "
             "Auto-loads voice WAV from voices/ and persona from personas/",
    )
    
    # Monitoring options
    parser.add_argument(
        "--no-memory-monitor",
        action="store_true",
        help="Disable memory usage monitoring",
    )
    
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=60,
        help="Memory monitor update interval in seconds (default: 60)",
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
    )

