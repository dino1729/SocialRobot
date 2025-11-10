"""Entrypoint for the basic voice assistant with wake word detection."""

from __future__ import annotations

import argparse
import os
import threading
import time
import struct
import numpy as np
from typing import Optional

import pyaudio
from dotenv import load_dotenv
from openwakeword.model import Model

from audio.stt import FasterWhisperSTT
from audio.tts import KokoroTTS
from audio.vad import VADConfig, VADListener
from llm.ollama import OllamaClient


# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")
WAKEWORD_MODEL = os.getenv("WAKEWORD_MODEL", "hey_jarvis_v0.1")  # Default wake word
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))  # Threshold for detection


def _get_memory_stats() -> dict[str, float]:
    """Get system memory statistics (lightweight, no external dependencies)."""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        mem_total = mem_available = mem_free = 0
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1]) / 1024  # Convert to MB
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
    """Get current process memory usage in MB."""
    try:
        pid = os.getpid()
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # Convert to MB
    except Exception:
        pass
    return 0.0


def _format_memory_stats(stats: dict[str, float], process_mem: float) -> str:
    """Format memory statistics for display."""
    return (
        f"💾 RAM: {stats['used']:.0f}/{stats['total']:.0f}MB "
        f"({stats['percent']:.1f}%) | "
        f"Process: {process_mem:.0f}MB"
    )


def _memory_monitor(stop_event: threading.Event, interval: int = 60) -> None:
    """Background thread to monitor and display memory usage."""
    while not stop_event.is_set():
        stats = _get_memory_stats()
        process_mem = _get_process_memory()
        print(f"\n{_format_memory_stats(stats, process_mem)}")
        stop_event.wait(interval)  # Sleep but can be interrupted


def _detect_whisper_device() -> str:
    """Detects the best available device for ctranslate2 (CUDA or CPU)."""
    try:
        import ctranslate2  # type: ignore

        if ctranslate2.get_cuda_device_count() > 0:  # type: ignore[attr-defined]
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _is_jetson() -> bool:
    """Detects if running on NVIDIA Jetson platform."""
    try:
        # Check for Jetson-specific hardware identifiers
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            return "jetson" in model or "tegra" in model
    except Exception:
        pass
    
    # Fallback: check for Jetson-specific environment
    return os.path.exists("/etc/nv_tegra_release")


def _get_default_compute_type(device: str) -> str:
    """Returns default compute type based on device and platform.
    
    Args:
        device: Device type ('cuda' or 'cpu')
    
    Returns:
        Default compute type ('int8', 'float16', or 'float32')
    """
    if device == "cpu":
        return "int8"  # CPU always uses int8
    
    # For CUDA, check if running on Jetson
    if _is_jetson():
        return "int8"  # Jetson Orin Nano uses int8
    
    # For desktop GPUs (RTX 5090, etc.), use float16 for better compatibility
    return "float16"


class WakeWordDetector:
    """Wake word detector using OpenWakeWord."""
    
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
            wakeword_models: List of wake word model names to load
            threshold: Detection threshold (0.0 to 1.0)
            chunk_size: Audio chunk size in samples
            sample_rate: Audio sample rate in Hz
            device_index: PyAudio device index (None for default)
        """
        self.wakeword_models = wakeword_models
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.device_index = device_index
        
        # Initialize OpenWakeWord model
        print(f"-> Loading wake word model(s): {', '.join(wakeword_models)}")
        self.oww_model = Model(wakeword_models=wakeword_models, inference_framework="onnx")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_running = False
        self.detection_callback: Optional[callable] = None
    
    def set_detection_callback(self, callback: callable) -> None:
        """Set callback function to be called when wake word is detected."""
        self.detection_callback = callback
    
    def start(self) -> None:
        """Start listening for wake word."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )
        
        print(f"-> Wake word detector ready! Say '{self.wakeword_models[0].replace('_', ' ')}' to activate...")
        
        # Start listening loop
        while self.is_running:
            try:
                # Read audio chunk
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Get predictions
                predictions = self.oww_model.predict(audio_array)
                
                # Check for wake word detection
                for model_name, score in predictions.items():
                    if score >= self.threshold:
                        print(f"🎤 Wake word detected! (confidence: {score:.2f})")
                        if self.detection_callback:
                            self.detection_callback()
                        break
                
            except Exception as e:
                if self.is_running:
                    print(f"Wake word detection error: {e}")
                break
    
    def stop(self) -> None:
        """Stop listening for wake word."""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.audio.terminate()


def main(
    enable_memory_monitor: bool = True,
    monitor_interval: int = 60,
    compute_type: Optional[str] = None,
    wakeword_threshold: float = WAKEWORD_THRESHOLD,
) -> None:
    """Initializes all components and starts the main interaction loop.
    
    Args:
        enable_memory_monitor: Whether to show periodic memory usage stats (default: True)
        monitor_interval: Seconds between memory stat updates (default: 60)
        compute_type: Compute type for faster-whisper ('int8', 'float16', or 'float32').
                     If None, auto-detects based on device and platform.
        wakeword_threshold: Detection threshold for wake word (0.0 to 1.0)
    """
    print("-> Initializing voice assistant with wake word detection...")
    print(f"-> Ollama URL: {OLLAMA_URL}")
    print(f"-> Ollama Model: {OLLAMA_MODEL}")
    print(f"-> Wake Word: {WAKEWORD_MODEL}")
    print(f"-> Wake Word Threshold: {wakeword_threshold}")
    
    # Display initial memory stats
    if enable_memory_monitor:
        stats = _get_memory_stats()
        process_mem = _get_process_memory()
        print(f"-> Initial {_format_memory_stats(stats, process_mem)}")
    
    # Start background memory monitor if enabled
    memory_stop_event = threading.Event()
    memory_thread = None
    if enable_memory_monitor:
        memory_thread = threading.Thread(
            target=_memory_monitor,
            args=(memory_stop_event, monitor_interval),
            daemon=True
        )
        memory_thread.start()
    
    vad_config = VADConfig(sample_rate=16000, frame_duration_ms=30, padding_duration_ms=360, aggressiveness=2)

    stt_device = _detect_whisper_device()
    
    # Determine compute type: use provided value or auto-detect based on platform
    if compute_type is None:
        compute_type = _get_default_compute_type(stt_device)
        print(f"-> Auto-detected compute type: {compute_type} (device: {stt_device})")
    else:
        # Validate compute type
        valid_types = ["int8", "float16", "float32"]
        if compute_type not in valid_types:
            print(f"-> Warning: Invalid compute type '{compute_type}', using default")
            compute_type = _get_default_compute_type(stt_device)
    
    stt_model = FasterWhisperSTT(model_size_or_path="tiny.en", device=stt_device, compute_type=compute_type)

    ollama_client = OllamaClient(
        url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        stream=True,
        system_prompt="You are a world-class knowledgeable AI voice assistant, Orion, hosted on a Jetson Orin Nano Super. Your mission is to assist users with any questions or tasks they have on a wide range of topics. Use your knowledge, skills, and resources to provide accurate, relevant, and helpful responses. Please remember that you are a voice assistant and keep answers brief, concise and within 1-2 sentences, unless it's absolutely necessary to give a longer response. Be polite, friendly, and respectful in your interactions, and try to satisfy the user's needs as best as you can. Dont include any emojis or asterisks or any other formatting in your responses.",
    )

    # Use GPU for TTS if available (will fallback to CPU if not available)
    tts_model = KokoroTTS(voice="af_bella", speed=1.0, use_gpu=False)

    vad_listener: Optional[VADListener] = None
    last_bot_response: str = ""
    listening_active = threading.Event()

    def on_speech_detected(raw_bytes: bytes) -> None:
        """Callback function triggered when VAD detects speech."""
        nonlocal vad_listener, last_bot_response
        if vad_listener is None:
            return

        vad_listener.disable_vad()

        try:
            recognized_text = stt_model.run_stt(raw_bytes, sample_rate=vad_listener.sample_rate)
        except Exception as exc:  # pragma: no cover - defensive logging only
            print("STT error:", exc)
            recognized_text = ""

        print("-> User said:", recognized_text)

        if not recognized_text.strip():
            vad_listener.stop()
            listening_active.clear()
            return

        # Simple echo cancellation: ignore if user input matches the last bot response
        normalized_user = recognized_text.strip().lower()
        normalized_bot = last_bot_response.strip().lower()
        if normalized_user and normalized_bot:
            if (
                normalized_user == normalized_bot
                or normalized_user in normalized_bot
                or normalized_bot in normalized_user
            ):
                print("-> Ignoring self-echo from recent response.")
                vad_listener.stop()
                listening_active.clear()
                return

        llm_response = ollama_client.query(recognized_text)
        print("-> Bot response:", llm_response)
        
        if not llm_response.strip():
            vad_listener.stop()
            listening_active.clear()
            return

        try:
            audio_data = tts_model.synthesize(llm_response)
        except Exception as exc:  # pragma: no cover - defensive logging only
            print("TTS error:", exc)
            vad_listener.stop()
            listening_active.clear()
            return

        last_bot_response = llm_response

        # Play audio without amplitude callback (no face animation)
        tts_model.play_audio_with_amplitude(audio_data, amplitude_callback=None)
        
        # Display memory stats after interaction
        if enable_memory_monitor:
            stats = _get_memory_stats()
            process_mem = _get_process_memory()
            print(f"-> {_format_memory_stats(stats, process_mem)}")
        
        # Stop VAD listener after response
        vad_listener.stop()
        listening_active.clear()

    def on_wakeword_detected() -> None:
        """Callback when wake word is detected - start listening for command."""
        nonlocal vad_listener
        
        if listening_active.is_set():
            return  # Already listening
        
        listening_active.set()
        print("-> Listening for command...")
        
        # Create new VAD listener for this session
        vad_listener = VADListener(
            config=vad_config,
            device_index=None,
            on_speech_callback=on_speech_detected
        )
        vad_listener.start()

    # Initialize wake word detector
    wakeword_detector = WakeWordDetector(
        wakeword_models=[WAKEWORD_MODEL],
        threshold=wakeword_threshold,
        chunk_size=1280,
        sample_rate=16000,
        device_index=None,
    )
    wakeword_detector.set_detection_callback(on_wakeword_detected)

    print("-> Voice assistant ready with wake word detection!")
    if enable_memory_monitor:
        print(f"-> Memory monitoring enabled (updates every {monitor_interval}s)")
    print("-> Press Ctrl+C to stop.")
    
    try:
        wakeword_detector.start()
    except KeyboardInterrupt:
        print("\n-> Shutting down...")
    finally:
        wakeword_detector.stop()
        
        if vad_listener:
            vad_listener.stop()
        
        # Stop memory monitor thread
        if enable_memory_monitor:
            memory_stop_event.set()
            if memory_thread and memory_thread.is_alive():
                memory_thread.join(timeout=1)
        
        # Display final memory stats
        if enable_memory_monitor:
            stats = _get_memory_stats()
            process_mem = _get_process_memory()
            print(f"-> Final {_format_memory_stats(stats, process_mem)}")
        
        print("-> Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voice assistant with wake word detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (auto-detects compute type)
  python main_wakeword.py
  
  # Use int8 compute type (for Jetson Orin Nano)
  python main_wakeword.py --compute-type int8
  
  # Use float16 compute type (for desktop GPUs like RTX 5090)
  python main_wakeword.py --compute-type float16
  
  # Custom wake word threshold
  python main_wakeword.py --wakeword-threshold 0.6
  
  # Disable memory monitoring
  python main_wakeword.py --no-memory-monitor
  
  # Change memory monitor interval
  python main_wakeword.py --monitor-interval 30
        """,
    )
    
    parser.add_argument(
        "--compute-type",
        type=str,
        choices=["int8", "float16", "float32"],
        default=None,
        help="Compute type for faster-whisper. Options: int8 (Jetson), float16 (desktop GPUs), float32. "
             "If not specified, auto-detects based on device and platform.",
    )
    
    parser.add_argument(
        "--wakeword-threshold",
        type=float,
        default=WAKEWORD_THRESHOLD,
        help=f"Wake word detection threshold (0.0 to 1.0, default: {WAKEWORD_THRESHOLD})",
    )
    
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
    
    args = parser.parse_args()
    
    main(
        enable_memory_monitor=not args.no_memory_monitor,
        monitor_interval=args.monitor_interval,
        compute_type=args.compute_type,
        wakeword_threshold=args.wakeword_threshold,
    )

