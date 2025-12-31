#!/usr/bin/env python3
"""
Quick STT (Speech-to-Text) Test Script
=======================================

Records audio from your microphone and transcribes it using the configured STT engine.
Uses settings from .env file or defaults.

Usage:
    python test_stt.py              # Use VAD (auto-detect speech)
    python test_stt.py --fixed 5    # Record for 5 seconds
    python test_stt.py --help       # Show options

Environment Variables (from .env):
    STT_ENGINE      - 'faster-whisper' or 'openai-whisper' (default: faster-whisper)
    STT_DEVICE      - 'cpu' or 'cuda' (auto-detected if not set)
    STT_MODEL       - Model size: tiny, base, small, medium, large (default: tiny)
    VAD_AGGRESSIVENESS - 0-3, higher = stricter noise filtering (default: 2)
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional

# Suppress ALSA error messages (they're harmless but noisy)
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                          ctypes.c_char_p, ctypes.c_int,
                                          ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass


@contextmanager
def suppress_stderr():
    """Suppress stderr output from C libraries (JACK, etc.)."""
    # Flush any pending stderr output
    sys.stderr.flush()
    
    # Save the original stderr file descriptor
    old_stderr_fd = os.dup(2)
    
    # Open /dev/null and redirect stderr to it
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    
    try:
        yield
    finally:
        # Restore stderr
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)


with suppress_stderr():
    import pyaudio

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


def detect_device() -> str:
    """Auto-detect best device (cuda or cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_compute_type(device: str) -> str:
    """Get optimal compute type for device."""
    if device == "cuda":
        return "float16"
    return "int8"


def record_audio_vad(
    sample_rate: int = 16000,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
    padding_duration_ms: int = 360,
    activation_ratio: float = 0.6,
    deactivation_ratio: float = 0.85,
    timeout_seconds: float = 30.0,
) -> Optional[bytes]:
    """Record audio using VAD (Voice Activity Detection).
    
    Returns audio bytes when speech is detected and ends.
    """
    import webrtcvad
    import collections

    vad = webrtcvad.Vad(aggressiveness)
    
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    padding_frames = max(1, int(padding_duration_ms / frame_duration_ms))
    activation_count = max(1, int(padding_frames * activation_ratio))
    deactivation_count = max(1, int(padding_frames * deactivation_ratio))
    
    # Suppress JACK warnings during stream creation
    with suppress_stderr():
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=frame_size,
        )
    
    print("🎤 Listening... (speak now, will auto-detect when you stop)")
    print("   Press Ctrl+C to cancel\n")
    
    ring_buffer = collections.deque(maxlen=padding_frames)
    triggered = False
    voiced_frames = []
    start_time = time.time()
    
    try:
        while True:
            if time.time() - start_time > timeout_seconds:
                print("⏱️  Timeout - no speech detected")
                return None
            
            frame = stream.read(frame_size, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, sample_rate)
            ring_buffer.append((frame, is_speech))
            
            if not triggered:
                num_voiced = sum(1 for _, speech in ring_buffer if speech)
                if len(ring_buffer) == ring_buffer.maxlen and num_voiced >= activation_count:
                    triggered = True
                    print("   🔊 Speech detected, recording...")
                    voiced_frames.extend([f for f, _ in ring_buffer])
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                num_unvoiced = sum(1 for _, speech in ring_buffer if not speech)
                if len(ring_buffer) == ring_buffer.maxlen and num_unvoiced >= deactivation_count:
                    print("   ✓ Speech ended\n")
                    break
    
    except KeyboardInterrupt:
        print("\n   Cancelled by user")
        return None
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
    
    return b"".join(voiced_frames)


def record_audio_fixed(seconds: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Record audio for a fixed duration."""
    # Suppress JACK warnings during stream creation
    with suppress_stderr():
        pa = pyaudio.PyAudio()
        frame_size = 1024
        
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=frame_size,
        )
    
    print(f"🎤 Recording for {seconds} seconds...")
    print("   Speak now!\n")
    
    frames = []
    frames_needed = int(sample_rate * seconds / frame_size)
    
    try:
        for i in range(frames_needed):
            frame = stream.read(frame_size, exception_on_overflow=False)
            frames.append(frame)
            
            # Progress indicator
            progress = (i + 1) / frames_needed
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '#' * filled + '-' * (bar_len - filled)
            remaining = seconds * (1 - progress)
            sys.stdout.write(f"\r   [{bar}] {remaining:.1f}s remaining")
            sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("\n   Cancelled by user")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
    
    print("\n   ✓ Recording complete\n")
    return b"".join(frames)


def calculate_audio_level(audio_bytes: bytes) -> float:
    """Calculate RMS level of audio in dB."""
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if len(audio) == 0:
        return -100.0
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1:
        return -100.0
    db = 20 * np.log10(rms / 32768.0)
    return db


def main():
    parser = argparse.ArgumentParser(
        description="Quick STT test - record and transcribe speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_stt.py              # Auto-detect speech with VAD
  python test_stt.py --fixed 5    # Record for 5 seconds
  python test_stt.py --engine openai-whisper
  python test_stt.py --model base # Use larger model for better accuracy
        """
    )
    parser.add_argument(
        "--fixed", "-f", type=float, metavar="SECONDS",
        help="Record for fixed duration instead of using VAD"
    )
    parser.add_argument(
        "--engine", "-e", type=str,
        choices=["faster-whisper", "openai-whisper"],
        default=os.getenv("STT_ENGINE", "faster-whisper"),
        help="STT engine to use (default: from .env or faster-whisper)"
    )
    parser.add_argument(
        "--device", "-d", type=str,
        choices=["cpu", "cuda"],
        default=os.getenv("STT_DEVICE"),
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--model", "-m", type=str,
        default=os.getenv("STT_MODEL", "tiny"),
        help="Model size: tiny, base, small, medium, large (default: tiny)"
    )
    parser.add_argument(
        "--vad-aggressiveness", type=int,
        default=int(os.getenv("VAD_AGGRESSIVENESS", "2")),
        choices=[0, 1, 2, 3],
        help="VAD aggressiveness 0-3 (default: 2)"
    )
    parser.add_argument(
        "--loop", "-l", action="store_true",
        help="Loop continuously (Ctrl+C to exit)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    device = args.device or detect_device()
    compute_type = get_compute_type(device)
    
    print("=" * 60)
    print("STT Test - Speech to Text")
    print("=" * 60)
    print()
    
    # Check audio configuration
    print("Checking audio configuration...")
    try:
        from audio.audio_devices import ensure_correct_audio_devices
        ensure_correct_audio_devices(verbose=True)
    except ImportError:
        print("   (audio_devices module not available, using system defaults)")
    print()
    
    # Initialize STT engine
    print(f"Loading STT engine...")
    print(f"   Engine: {args.engine}")
    print(f"   Model:  {args.model}")
    print(f"   Device: {device}")
    print(f"   Compute: {compute_type}")
    print()
    
    try:
        from audio.engine_config import create_stt_engine
        stt = create_stt_engine(
            engine=args.engine,
            device=device,
            model_size=args.model,
            compute_type=compute_type,
            language="en",
        )
    except Exception as e:
        print(f"❌ Failed to load STT engine: {e}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    
    sample_rate = 16000
    
    while True:
        print()
        
        # Record audio
        if args.fixed:
            audio_bytes = record_audio_fixed(args.fixed, sample_rate)
        else:
            audio_bytes = record_audio_vad(
                sample_rate=sample_rate,
                aggressiveness=args.vad_aggressiveness,
            )
        
        if not audio_bytes:
            if args.loop:
                continue
            break
        
        # Calculate audio stats
        duration = len(audio_bytes) / (sample_rate * 2)  # 16-bit = 2 bytes per sample
        level_db = calculate_audio_level(audio_bytes)
        
        print(f"📊 Audio Stats:")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Level: {level_db:.1f} dB")
        print(f"   Size: {len(audio_bytes):,} bytes")
        print()
        
        # Transcribe
        print("🔄 Transcribing...")
        start_time = time.time()
        
        try:
            text = stt.run_stt(audio_bytes, sample_rate=sample_rate)
            elapsed = time.time() - start_time
            
            print(f"⏱️  Transcription time: {elapsed:.2f}s")
            print()
            print("=" * 60)
            print("📝 TRANSCRIPTION:")
            print("=" * 60)
            if text:
                print(f"\n   \"{text}\"\n")
            else:
                print("\n   (no speech detected or empty transcription)\n")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
        
        if not args.loop:
            break
        
        print("\n[Press Ctrl+C to exit, or speak again to continue...]\n")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
