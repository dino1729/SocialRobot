#!/usr/bin/env python3
"""
Quick TTS (Text-to-Speech) Test Script
=======================================

Converts text to speech using the configured TTS engine and plays it.
Uses settings from .env file or defaults.

Usage:
    python test_tts.py "Hello, world!"
    python test_tts.py --engine chatterbox --gpu "Testing voice cloning"
    python test_tts.py --list-voices
    python test_tts.py --help

Environment Variables (from .env):
    TTS_ENGINE      - 'kokoro', 'piper', 'chatterbox', or 'vibevoice' (default: kokoro)
    TTS_GPU         - Enable GPU acceleration (default: false)
    TTS_VOICE       - Voice character for Chatterbox (e.g., morgan_freeman)
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# Suppress ALSA error messages
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


def list_available_voices():
    """List available voices for each TTS engine."""
    script_dir = Path(__file__).parent
    
    print("=" * 60)
    print("Available TTS Voices")
    print("=" * 60)
    
    # Kokoro voices
    print("\n📢 KOKORO (CPU-based, fast)")
    print("-" * 40)
    try:
        from audio.tts import KokoroTTS
        tts = KokoroTTS()
        voices = tts.available_voices()
        for i, voice in enumerate(voices[:20]):  # Limit to 20
            print(f"   {voice}")
        if len(voices) > 20:
            print(f"   ... and {len(voices) - 20} more")
    except Exception as e:
        print(f"   (unavailable: {e})")
    
    # Chatterbox voices (from voices/ directory)
    print("\n📢 CHATTERBOX (zero-shot voice cloning)")
    print("-" * 40)
    voices_dir = script_dir / "voices"
    if voices_dir.exists():
        wav_files = sorted(voices_dir.glob("*.wav"))
        if wav_files:
            for wav in wav_files:
                voice_name = wav.stem
                persona_exists = (script_dir / "personas" / f"{voice_name}.txt").exists()
                marker = "✓" if persona_exists else " "
                print(f"   {marker} {voice_name}")
            print("\n   ✓ = has matching persona file")
        else:
            print("   (no voice files in voices/ directory)")
    else:
        print("   (voices/ directory not found)")
    
    # VibeVoice speakers
    print("\n📢 VIBEVOICE (neural TTS)")
    print("-" * 40)
    vibevoice_speakers = [
        "Carter", "Bria", "Alex", "Dora", "Nova", 
        "Sol", "Aria", "Isla", "Eva", "Maya", "Raj"
    ]
    for speaker in vibevoice_speakers:
        print(f"   {speaker}")
    
    # Piper
    print("\n📢 PIPER (lightweight local TTS)")
    print("-" * 40)
    print("   Uses system-installed Piper models")
    print("   Check ~/.local/share/piper-tts/ for installed voices")
    
    print()


def play_audio(audio_data: np.ndarray, sample_rate: int):
    """Play audio data through the default speaker."""
    if audio_data is None or len(audio_data) == 0:
        print("   (no audio to play)")
        return
    
    # Convert to int16 if needed
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        audio_int16 = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
    else:
        audio_int16 = audio_data.astype(np.int16)
    
    with suppress_stderr():
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=4096,
        )
    
    try:
        stream.write(audio_int16.tobytes())
        # Wait for playback to complete
        time.sleep(0.1)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="Quick TTS test - convert text to speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_tts.py "Hello, world!"
  python test_tts.py --engine chatterbox --voice morgan_freeman "Hello there!"
  python test_tts.py --engine vibevoice --speaker Bria "Testing neural TTS"
  python test_tts.py --engine kokoro --voice af_sarah "Different voice"
  python test_tts.py --gpu "Using GPU acceleration"
  python test_tts.py --list-voices
  python test_tts.py --save output.wav "Save to file"
        """
    )
    parser.add_argument(
        "text", nargs="*",
        help="Text to convert to speech"
    )
    parser.add_argument(
        "--engine", "-e", type=str,
        choices=["kokoro", "piper", "chatterbox", "vibevoice"],
        default=os.getenv("TTS_ENGINE", "kokoro"),
        help="TTS engine to use (default: from .env or kokoro)"
    )
    parser.add_argument(
        "--gpu", "-g", action="store_true",
        default=get_env_bool("TTS_GPU", False),
        help="Enable GPU acceleration"
    )
    parser.add_argument(
        "--voice", "-v", type=str,
        default=os.getenv("TTS_VOICE"),
        help="Voice for Kokoro (e.g., af_bella) or Chatterbox character (e.g., morgan_freeman)"
    )
    parser.add_argument(
        "--speaker", "-s", type=str,
        default=os.getenv("VIBEVOICE_SPEAKER", "Carter"),
        help="Speaker for VibeVoice (default: Carter)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speech speed for Kokoro (default: 1.0)"
    )
    parser.add_argument(
        "--save", type=str, metavar="FILE",
        help="Save audio to WAV file instead of playing"
    )
    parser.add_argument(
        "--list-voices", "-l", action="store_true",
        help="List available voices for each engine"
    )
    parser.add_argument(
        "--no-play", action="store_true",
        help="Don't play audio (useful with --save)"
    )
    
    args = parser.parse_args()
    
    # List voices mode
    if args.list_voices:
        list_available_voices()
        return
    
    # Check for text input
    if not args.text:
        parser.print_help()
        print("\n❌ Error: Please provide text to convert to speech")
        print('   Example: python test_tts.py "Hello, world!"')
        sys.exit(1)
    
    text = " ".join(args.text)
    
    print("=" * 60)
    print("TTS Test - Text to Speech")
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
    
    # Show configuration
    print(f"Configuration:")
    print(f"   Engine:  {args.engine}")
    print(f"   GPU:     {'Enabled' if args.gpu else 'Disabled'}")
    if args.voice:
        print(f"   Voice:   {args.voice}")
    if args.engine == "vibevoice":
        print(f"   Speaker: {args.speaker}")
    if args.engine == "kokoro":
        print(f"   Speed:   {args.speed}")
    print()
    
    # Build TTS engine arguments
    print("Loading TTS engine...")
    start_load = time.time()
    
    try:
        from audio.engine_config import create_tts_engine
        
        kwargs = {
            "engine": args.engine,
            "use_gpu": args.gpu,
        }
        
        if args.engine == "kokoro":
            if args.voice:
                kwargs["voice"] = args.voice
            kwargs["speed"] = args.speed
            
        elif args.engine == "chatterbox":
            if args.voice:
                # Look for voice file in voices/ directory
                script_dir = Path(__file__).parent
                voice_path = script_dir / "voices" / f"{args.voice}.wav"
                if voice_path.exists():
                    kwargs["voice_path"] = str(voice_path)
                    print(f"   Using voice file: {voice_path}")
                else:
                    print(f"   ⚠️  Voice file not found: {voice_path}")
                    print(f"   Available voices: {', '.join(p.stem for p in (script_dir / 'voices').glob('*.wav'))}")
                    
        elif args.engine == "vibevoice":
            kwargs["speaker"] = args.speaker
        
        tts = create_tts_engine(**kwargs)
        load_time = time.time() - start_load
        print(f"   ✓ Loaded in {load_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Failed to load TTS engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("=" * 60)
    print(f"📝 Input Text:")
    print(f"   \"{text}\"")
    print("=" * 60)
    print()
    
    # Synthesize speech
    print("🔊 Synthesizing speech...")
    start_synth = time.time()
    
    try:
        audio_data = tts.synthesize(text)
        synth_time = time.time() - start_synth
        
        # Get sample rate
        sample_rate = getattr(tts, 'sample_rate', 22050)
        
        # Calculate audio duration
        duration = len(audio_data) / sample_rate if len(audio_data) > 0 else 0
        
        print(f"   ✓ Synthesized in {synth_time:.2f}s")
        print()
        print(f"📊 Audio Stats:")
        print(f"   Duration:    {duration:.2f}s")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Samples:     {len(audio_data):,}")
        print(f"   RTF:         {synth_time/duration:.2f}x" if duration > 0 else "   RTF: N/A")
        print()
        
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save to file if requested
    if args.save:
        try:
            import wave
            
            # Convert to int16
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_int16 = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            with wave.open(args.save, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            print(f"💾 Saved to: {args.save}")
            print()
        except Exception as e:
            print(f"⚠️  Failed to save audio: {e}")
    
    # Play audio
    if not args.no_play and len(audio_data) > 0:
        print("🔈 Playing audio...")
        play_audio(audio_data, sample_rate)
        print("   ✓ Playback complete")
    
    print()
    print("=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
