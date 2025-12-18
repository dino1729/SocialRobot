#!/usr/bin/env python3
"""
TTS Engine Benchmark Script
===========================
Compares Chatterbox TTS vs VibeVoice TTS performance on the same text.

Metrics measured:
- Total synthesis time
- Time to first audio (if applicable)
- GPU memory consumption (peak and average)
- Audio duration and Real-Time Factor (RTF)
- Audio quality metrics (sample rate, file size)

Usage:
    python benchmark_tts_engines.py
    python benchmark_tts_engines.py --text "Custom text to synthesize"
    python benchmark_tts_engines.py --runs 5  # Multiple runs for averaging
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

# Check for torch and GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, GPU monitoring disabled")


@dataclass
class BenchmarkResult:
    """Results from a single TTS benchmark run."""
    engine_name: str
    text_length: int
    synthesis_time: float
    audio_duration: float
    rtf: float  # Real-Time Factor (synthesis_time / audio_duration)
    sample_rate: int
    audio_samples: int
    gpu_memory_peak_mb: float
    gpu_memory_allocated_mb: float
    success: bool
    error: Optional[str] = None


class GPUMemoryMonitor:
    """Monitor GPU memory usage during TTS synthesis."""

    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
        self.available = TORCH_AVAILABLE and torch.cuda.is_available()

    def reset(self):
        """Reset memory tracking."""
        if self.available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()

    def get_peak_mb(self) -> float:
        """Get peak memory usage in MB."""
        if self.available:
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    def get_allocated_mb(self) -> float:
        """Get current allocated memory in MB."""
        if self.available:
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def get_used_mb(self) -> float:
        """Get memory used since reset in MB."""
        if self.available:
            return (torch.cuda.memory_allocated() - self.start_memory) / (1024 * 1024)
        return 0.0


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_chatterbox(
    text: str,
    voice_path: str,
    gpu_monitor: GPUMemoryMonitor,
) -> BenchmarkResult:
    """Benchmark Chatterbox TTS."""
    print("\n" + "=" * 60)
    print("Benchmarking: Chatterbox TTS (Rick Sanchez voice)")
    print("=" * 60)

    clear_gpu_memory()

    try:
        from audio.tts_chatterbox import ChatterboxTTS

        # Initialize model
        print("-> Loading Chatterbox model...")
        init_start = time.time()
        tts = ChatterboxTTS(voice_path=voice_path, use_gpu=True)
        init_time = time.time() - init_start
        print(f"-> Model loaded in {init_time:.2f}s")

        # Reset GPU memory tracking after model load
        gpu_monitor.reset()

        # Synthesize
        print(f"-> Synthesizing {len(text)} characters...")
        synth_start = time.time()
        audio = tts.synthesize(text)
        synth_time = time.time() - synth_start

        # Get GPU stats
        peak_memory = gpu_monitor.get_peak_mb()
        allocated_memory = gpu_monitor.get_allocated_mb()

        # Calculate metrics
        audio_duration = len(audio) / tts.sample_rate if len(audio) > 0 else 0
        rtf = synth_time / audio_duration if audio_duration > 0 else float('inf')

        print(f"-> Synthesis complete: {audio_duration:.2f}s audio in {synth_time:.2f}s")
        print(f"-> RTF: {rtf:.2f}x (lower is better)")
        print(f"-> GPU Peak Memory: {peak_memory:.1f} MB")

        # Save audio
        output_path = "benchmark_chatterbox.wav"
        import scipy.io.wavfile as wav
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wav.write(output_path, tts.sample_rate, audio_int16)
        print(f"-> Saved to: {output_path}")

        # Cleanup
        del tts
        clear_gpu_memory()

        return BenchmarkResult(
            engine_name="Chatterbox TTS",
            text_length=len(text),
            synthesis_time=synth_time,
            audio_duration=audio_duration,
            rtf=rtf,
            sample_rate=24000,
            audio_samples=len(audio),
            gpu_memory_peak_mb=peak_memory,
            gpu_memory_allocated_mb=allocated_memory,
            success=True,
        )

    except Exception as e:
        print(f"-> ERROR: {e}")
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            engine_name="Chatterbox TTS",
            text_length=len(text),
            synthesis_time=0,
            audio_duration=0,
            rtf=0,
            sample_rate=0,
            audio_samples=0,
            gpu_memory_peak_mb=0,
            gpu_memory_allocated_mb=0,
            success=False,
            error=str(e),
        )


def benchmark_vibevoice(
    text: str,
    speaker: str,
    gpu_monitor: GPUMemoryMonitor,
) -> BenchmarkResult:
    """Benchmark VibeVoice TTS."""
    print("\n" + "=" * 60)
    print(f"Benchmarking: VibeVoice TTS ({speaker} voice)")
    print("=" * 60)

    clear_gpu_memory()

    try:
        from audio.tts_vibevoice import VibeVoiceTTS

        # Initialize model
        print("-> Loading VibeVoice model...")
        init_start = time.time()
        tts = VibeVoiceTTS(speaker=speaker, use_gpu=True)
        init_time = time.time() - init_start
        print(f"-> Model loaded in {init_time:.2f}s")

        # Reset GPU memory tracking after model load
        gpu_monitor.reset()

        # Synthesize
        print(f"-> Synthesizing {len(text)} characters...")
        synth_start = time.time()
        audio = tts.synthesize(text)
        synth_time = time.time() - synth_start

        # Get GPU stats
        peak_memory = gpu_monitor.get_peak_mb()
        allocated_memory = gpu_monitor.get_allocated_mb()

        # Calculate metrics
        audio_duration = len(audio) / tts.sample_rate if len(audio) > 0 else 0
        rtf = synth_time / audio_duration if audio_duration > 0 else float('inf')

        print(f"-> Synthesis complete: {audio_duration:.2f}s audio in {synth_time:.2f}s")
        print(f"-> RTF: {rtf:.2f}x (lower is better)")
        print(f"-> GPU Peak Memory: {peak_memory:.1f} MB")

        # Save audio
        output_path = "benchmark_vibevoice.wav"
        import scipy.io.wavfile as wav
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wav.write(output_path, tts.sample_rate, audio_int16)
        print(f"-> Saved to: {output_path}")

        # Cleanup
        del tts
        clear_gpu_memory()

        return BenchmarkResult(
            engine_name="VibeVoice TTS",
            text_length=len(text),
            synthesis_time=synth_time,
            audio_duration=audio_duration,
            rtf=rtf,
            sample_rate=24000,
            audio_samples=len(audio),
            gpu_memory_peak_mb=peak_memory,
            gpu_memory_allocated_mb=allocated_memory,
            success=True,
        )

    except Exception as e:
        print(f"-> ERROR: {e}")
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            engine_name="VibeVoice TTS",
            text_length=len(text),
            synthesis_time=0,
            audio_duration=0,
            rtf=0,
            sample_rate=0,
            audio_samples=0,
            gpu_memory_peak_mb=0,
            gpu_memory_allocated_mb=0,
            success=False,
            error=str(e),
        )


def print_comparison(results: List[BenchmarkResult]):
    """Print a comparison table of benchmark results."""
    print("\n")
    print("=" * 80)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 80)

    # Filter successful results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        print("No successful benchmarks to compare.")
        return

    # Print table header
    print(f"\n{'Metric':<30} ", end="")
    for r in successful:
        print(f"{r.engine_name:<20} ", end="")
    print()
    print("-" * (30 + 21 * len(successful)))

    # Print metrics
    metrics = [
        ("Text Length (chars)", lambda r: f"{r.text_length}"),
        ("Synthesis Time (s)", lambda r: f"{r.synthesis_time:.2f}"),
        ("Audio Duration (s)", lambda r: f"{r.audio_duration:.2f}"),
        ("Real-Time Factor", lambda r: f"{r.rtf:.2f}x"),
        ("Speed vs Realtime", lambda r: f"{1/r.rtf:.1f}x faster" if r.rtf > 0 else "N/A"),
        ("Sample Rate (Hz)", lambda r: f"{r.sample_rate}"),
        ("GPU Peak Memory (MB)", lambda r: f"{r.gpu_memory_peak_mb:.1f}"),
        ("GPU Allocated (MB)", lambda r: f"{r.gpu_memory_allocated_mb:.1f}"),
    ]

    for metric_name, metric_fn in metrics:
        print(f"{metric_name:<30} ", end="")
        for r in successful:
            print(f"{metric_fn(r):<20} ", end="")
        print()

    # Print winner analysis
    print("\n" + "-" * 80)
    print("ANALYSIS:")
    print("-" * 80)

    if len(successful) >= 2:
        # Speed comparison
        fastest = min(successful, key=lambda r: r.rtf)
        slowest = max(successful, key=lambda r: r.rtf)
        speed_diff = slowest.rtf / fastest.rtf if fastest.rtf > 0 else 0
        print(f"  Speed Winner: {fastest.engine_name} ({speed_diff:.1f}x faster than {slowest.engine_name})")

        # Memory comparison
        lowest_mem = min(successful, key=lambda r: r.gpu_memory_peak_mb)
        highest_mem = max(successful, key=lambda r: r.gpu_memory_peak_mb)
        mem_diff = highest_mem.gpu_memory_peak_mb - lowest_mem.gpu_memory_peak_mb
        print(f"  Memory Winner: {lowest_mem.engine_name} ({mem_diff:.0f} MB less than {highest_mem.engine_name})")

    # Print failed benchmarks
    if failed:
        print("\n" + "-" * 80)
        print("FAILED BENCHMARKS:")
        for r in failed:
            print(f"  {r.engine_name}: {r.error}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TTS engines (Chatterbox vs VibeVoice)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text to synthesize. If not provided, uses default paragraph.",
    )

    parser.add_argument(
        "--voice-path",
        type=str,
        default="voices/rick_sanchez.wav",
        help="Path to voice WAV for Chatterbox (default: voices/rick_sanchez.wav)",
    )

    parser.add_argument(
        "--vibevoice-speaker",
        type=str,
        default="en-carter_man",
        help="VibeVoice speaker name (default: en-carter_man)",
    )

    parser.add_argument(
        "--only",
        type=str,
        choices=["chatterbox", "vibevoice"],
        default=None,
        help="Only benchmark specific engine",
    )

    args = parser.parse_args()

    # Default benchmark text
    if args.text is None:
        text = """The universe is expanding, but not uniformly across all regions of space. Scientists have discovered that distant galaxies are receding from us faster than nearby ones, following what we call Hubble's Law. Even more puzzling, recent measurements suggest the expansion rate in the early universe differs from what we observe today. This discrepancy, known as the Hubble tension, remains one of the biggest mysteries in modern cosmology."""
    else:
        text = args.text

    print("=" * 80)
    print("TTS ENGINE BENCHMARK")
    print("=" * 80)
    print(f"\nText to synthesize ({len(text)} chars):")
    print(f"  \"{text[:100]}...\"" if len(text) > 100 else f"  \"{text}\"")

    # Check GPU availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem:.1f} GB")
    else:
        print("\nWarning: No GPU available, benchmarks will run on CPU")

    # Initialize GPU monitor
    gpu_monitor = GPUMemoryMonitor()

    # Run benchmarks
    results = []

    if args.only != "vibevoice":
        # Check if voice file exists
        if os.path.exists(args.voice_path):
            result = benchmark_chatterbox(text, args.voice_path, gpu_monitor)
            results.append(result)
        else:
            print(f"\nWarning: Chatterbox voice file not found: {args.voice_path}")
            print("Skipping Chatterbox benchmark. Create a voice file or use --voice-path")

    if args.only != "chatterbox":
        result = benchmark_vibevoice(text, args.vibevoice_speaker, gpu_monitor)
        results.append(result)

    # Print comparison
    print_comparison(results)

    # Print output files
    print("\nGenerated audio files:")
    for f in ["benchmark_chatterbox.wav", "benchmark_vibevoice.wav"]:
        if os.path.exists(f):
            size_kb = os.path.getsize(f) / 1024
            print(f"  {f}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
