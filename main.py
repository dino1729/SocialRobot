"""Entrypoint for the basic voice assistant."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv

from audio.stt import FasterWhisperSTT
from audio.tts import KokoroTTS
from audio.vad import VADConfig, VADListener
from llm.ollama import OllamaClient


# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")


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


def main(enable_memory_monitor: bool = True, monitor_interval: int = 60) -> None:
    """Initializes all components and starts the main interaction loop.
    
    Args:
        enable_memory_monitor: Whether to show periodic memory usage stats (default: True)
        monitor_interval: Seconds between memory stat updates (default: 60)
    """
    print("-> Initializing basic voice assistant...")
    print(f"-> Ollama URL: {OLLAMA_URL}")
    print(f"-> Ollama Model: {OLLAMA_MODEL}")
    
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
    stt_model = FasterWhisperSTT(model_size_or_path="tiny.en", device=stt_device, compute_type="int8")

    ollama_client = OllamaClient(
        url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        stream=True,
        system_prompt="You are a cheerful robotic companion speaking concisely.",
    )

    tts_model = KokoroTTS(voice="af_bella", speed=1.0)

    vad_listener: Optional[VADListener] = None
    last_bot_response: str = ""

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
            vad_listener.enable_vad()
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
                vad_listener.enable_vad()
                return

        llm_response = ollama_client.query(recognized_text)
        print("-> Bot response:", llm_response)
        
        if not llm_response.strip():
            vad_listener.enable_vad()
            return

        try:
            audio_data = tts_model.synthesize(llm_response)
        except Exception as exc:  # pragma: no cover - defensive logging only
            print("TTS error:", exc)
            vad_listener.enable_vad()
            return

        last_bot_response = llm_response

        # Play audio without amplitude callback (no face animation)
        tts_model.play_audio_with_amplitude(audio_data, amplitude_callback=None)
        
        # Display memory stats after interaction
        if enable_memory_monitor:
            stats = _get_memory_stats()
            process_mem = _get_process_memory()
            print(f"-> {_format_memory_stats(stats, process_mem)}")
        
        vad_listener.enable_vad()

    vad_listener = VADListener(config=vad_config, device_index=None, on_speech_callback=on_speech_detected)

    print("-> Voice assistant ready! Listening for speech...")
    if enable_memory_monitor:
        print(f"-> Memory monitoring enabled (updates every {monitor_interval}s)")
    print("-> Press Ctrl+C to stop.")
    
    try:
        vad_listener.start()
    except KeyboardInterrupt:
        print("\n-> Shutting down...")
    finally:
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
    # Default: memory monitoring enabled with 60s interval
    main()
    
    # To disable memory monitoring, use:
    # main(enable_memory_monitor=False)
    
    # To change update interval (e.g., every 30 seconds):
    # main(enable_memory_monitor=True, monitor_interval=30)
