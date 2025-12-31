#!/usr/bin/env python3
"""
TTS Engine Benchmark Script
===========================
Benchmarks all available TTS engines on latency and quality (roundtrip WER).

Quality is measured by:
1. Synthesizing text with the TTS engine
2. Transcribing the audio with a judge STT (faster-whisper)
3. Computing WER between original text and transcription

Metrics measured:
- Init time
- Synthesis latency and RTF (real-time factor)
- Roundtrip WER (text -> TTS -> STT -> compare)
- GPU memory consumption (if available)

Usage:
    python benchmark_tts_engines.py
    python benchmark_tts_engines.py --no-gpu
    python benchmark_tts_engines.py --judge-model-size base
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from audio.engine_config import (
    STTEngine,
    TTSEngine,
    create_stt_engine,
    create_tts_engine,
    word_error_rate,
)
from benchmark import check_cuda_availability

# Check for torch and GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Benchmark phrases
# =============================================================================

TTS_BENCHMARK_PHRASES = [
    # Greeting and conversational
    "Hello there! How are you doing today? I hope you're having a wonderful day so far.",
    
    # Descriptive narrative
    "The quick brown fox jumped gracefully over the lazy dog, who was resting peacefully under the old oak tree. "
    "It was a perfect summer afternoon with a gentle breeze.",
    
    # Question with context
    "What's the weather forecast looking like for this weekend in San Francisco? "
    "I'm planning a trip to the Golden Gate Bridge and want to make sure it won't be too foggy.",
    
    # Multi-step instructions
    "First, please set a timer for fifteen minutes. Then, remind me to check my email and respond to the meeting invitation from the marketing team.",
    
    # Scientific and philosophical
    "The universe is incredibly vast and full of countless mysteries waiting to be discovered. "
    "Scientists estimate there are more stars in the cosmos than grains of sand on all of Earth's beaches.",
    
    # Helpful assistant response
    "Of course, I'd be happy to help you find a great restaurant nearby! "
    "Based on your preferences, I found three highly-rated Italian places within a ten-minute drive from your location.",
    
    # Numerical information
    "The conference call is scheduled for two forty-five on Wednesday, January fifteenth. "
    "There will be approximately thirty participants joining from offices in New York, London, and Tokyo.",
    
    # Technology explanation
    "Technology is advancing at an absolutely unprecedented rate, transforming how we live and work. "
    "From artificial intelligence to renewable energy, these innovations are reshaping our world in remarkable ways.",
    
    # Story-like narration
    "Once upon a time, in a small village nestled between rolling hills and a sparkling river, "
    "there lived a curious young inventor who dreamed of building machines that could help everyone.",
    
    # Emotional and expressive
    "I'm so excited to share this wonderful news with you! After months of hard work and dedication, "
    "the project has finally been approved, and we can begin implementation next week. Congratulations to the entire team!",
    
    # Bhagavad Gita - ancient Sanskrit terms (Chapter 2, Verse 47)
    "Karmanye vadhikaraste ma phaleshu kadachana. You have the right to perform your prescribed duties, "
    "but you are not entitled to the fruits of your actions. This is the teaching of Nishkama Karma, selfless action.",
    
    # Bhagavad Gita - philosophical (Chapter 2, Verse 20)
    "The Atman, the eternal soul, is neither born nor does it ever die. It is unborn, eternal, and primeval. "
    "This wisdom was imparted by Lord Krishna to Arjuna on the battlefield of Kurukshetra.",
]


@dataclass
class TTSBenchmarkResult:
    """Results from benchmarking a single TTS engine."""
    engine: str
    init_time: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    avg_rtf: float
    avg_roundtrip_wer: float
    total_phrases: int
    successful_phrases: int
    avg_audio_duration: float
    gpu_memory_peak_mb: float
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


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def audio_to_bytes(audio_data: np.ndarray) -> bytes:
    """Convert float32 audio to int16 bytes."""
    audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


def save_audio_wav(audio_data: np.ndarray, sample_rate: int, filepath: str) -> None:
    """Save audio as WAV file."""
    audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def benchmark_tts_engine(
    engine: TTSEngine,
    phrases: List[str],
    judge_stt: Any,
    use_gpu: bool = False,
    output_dir: Path = None,
    gpu_monitor: GPUMemoryMonitor = None,
    voice_path: Optional[str] = None,
    speaker: Optional[str] = None,
) -> TTSBenchmarkResult:
    """Benchmark a single TTS engine.
    
    Args:
        engine: TTS engine to benchmark
        phrases: List of phrases to synthesize
        judge_stt: STT engine for transcription (for roundtrip WER)
        use_gpu: Use GPU if available
        output_dir: Directory to save audio samples
        gpu_monitor: GPU memory monitor
        voice_path: Voice path for Chatterbox
        speaker: Speaker for VibeVoice
        
    Returns:
        TTSBenchmarkResult with timing and quality metrics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine.value}")
    print(f"{'='*60}")
    
    clear_gpu_memory()
    
    # Prepare output directory
    if output_dir:
        engine_output_dir = output_dir / engine.value
        engine_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TTS engine
    print("-> Initializing TTS engine...")
    try:
        init_start = time.time()
        
        # Engine-specific initialization
        kwargs = {"use_gpu": use_gpu}
        if engine == TTSEngine.CHATTERBOX:
            kwargs["voice_path"] = voice_path
            kwargs["use_turbo"] = True
        elif engine == TTSEngine.VIBEVOICE:
            kwargs["speaker"] = speaker or "wayne"
        elif engine == TTSEngine.KOKORO:
            kwargs["voice"] = "af_bella"
            kwargs["speed"] = 1.0
        
        tts = create_tts_engine(engine=engine, **kwargs)
        init_time = time.time() - init_start
        print(f"   Init time: {init_time:.3f}s")
        
    except Exception as e:
        print(f"ERROR: Failed to initialize {engine.value}: {e}")
        import traceback
        traceback.print_exc()
        return TTSBenchmarkResult(
            engine=engine.value,
            init_time=0.0,
            avg_latency=0.0,
            p50_latency=0.0,
            p95_latency=0.0,
            avg_rtf=0.0,
            avg_roundtrip_wer=1.0,
            total_phrases=len(phrases),
            successful_phrases=0,
            avg_audio_duration=0.0,
            gpu_memory_peak_mb=0.0,
            error=str(e),
        )
    
    sample_rate = getattr(tts, "sample_rate", 24000)
    
    # Reset GPU memory tracking
    if gpu_monitor:
        gpu_monitor.reset()
    
    # Benchmark synthesis
    latencies = []
    rtfs = []
    wers = []
    audio_durations = []
    successful = 0
    
    for i, phrase in enumerate(phrases):
        print(f"\n   [{i+1}/{len(phrases)}] \"{phrase[:40]}...\"")
        
        try:
            # Synthesize
            synth_start = time.time()
            audio_data = tts.synthesize(phrase)
            synth_time = time.time() - synth_start
            
            if audio_data is None or len(audio_data) == 0:
                print(f"      WARNING: Empty audio")
                continue
            
            # Calculate RTF
            audio_duration = len(audio_data) / sample_rate
            rtf = synth_time / audio_duration if audio_duration > 0 else float("inf")
            
            latencies.append(synth_time)
            rtfs.append(rtf)
            audio_durations.append(audio_duration)
            
            # Save audio sample
            if output_dir:
                audio_path = engine_output_dir / f"phrase_{i:03d}.wav"
                save_audio_wav(audio_data, sample_rate, str(audio_path))
            
            # Transcribe with judge STT for roundtrip WER
            audio_bytes = audio_to_bytes(audio_data)
            try:
                transcription = judge_stt.run_stt(audio_bytes, sample_rate=sample_rate)
                wer = word_error_rate(phrase, transcription)
                wers.append(wer)
                
                wer_pct = wer * 100
                print(f"      Latency={synth_time:.3f}s RTF={rtf:.2f}x WER={wer_pct:.1f}%")
                print(f"      Transcribed: \"{transcription[:50]}...\"")
                
            except Exception as e:
                print(f"      WARNING: Transcription failed: {e}")
                wers.append(1.0)
            
            successful += 1
            
        except Exception as e:
            print(f"      ERROR: {e}")
            continue
    
    # Get GPU memory peak
    peak_memory = gpu_monitor.get_peak_mb() if gpu_monitor else 0.0
    
    # Cleanup
    del tts
    clear_gpu_memory()
    
    # Aggregate results
    if not latencies:
        return TTSBenchmarkResult(
            engine=engine.value,
            init_time=init_time,
            avg_latency=0.0,
            p50_latency=0.0,
            p95_latency=0.0,
            avg_rtf=0.0,
            avg_roundtrip_wer=1.0,
            total_phrases=len(phrases),
            successful_phrases=0,
            avg_audio_duration=0.0,
            gpu_memory_peak_mb=peak_memory,
            error="No successful syntheses",
        )
    
    sorted_latencies = sorted(latencies)
    p50_idx = int(len(sorted_latencies) * 0.50)
    p95_idx = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)
    
    return TTSBenchmarkResult(
        engine=engine.value,
        init_time=init_time,
        avg_latency=statistics.mean(latencies),
        p50_latency=sorted_latencies[p50_idx],
        p95_latency=sorted_latencies[p95_idx],
        avg_rtf=statistics.mean(rtfs),
        avg_roundtrip_wer=statistics.mean(wers) if wers else 1.0,
        total_phrases=len(phrases),
        successful_phrases=successful,
        avg_audio_duration=statistics.mean(audio_durations),
        gpu_memory_peak_mb=peak_memory,
    )


def print_tts_results_table(results: List[TTSBenchmarkResult], output_dir: Path) -> None:
    """Print and save TTS benchmark results table with comprehensive summary."""
    print("\n")
    print("╔" + "═" * 108 + "╗")
    print("║" + " TTS ENGINE BENCHMARK RESULTS ".center(108) + "║")
    print("╚" + "═" * 108 + "╝")
    
    # Filter successful results
    valid = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    
    if not valid:
        print("No successful benchmark results.")
        return
    
    # Sort by quality (roundtrip WER)
    quality_sorted = sorted(valid, key=lambda r: r.avg_roundtrip_wer)
    
    # Print detailed results table
    print("\n┌" + "─" * 108 + "┐")
    print("│" + " DETAILED RESULTS (sorted by Roundtrip WER) ".center(108) + "│")
    print("├" + "─" * 108 + "┤")
    
    header = (
        f"│ {'Rank':<4} │ {'Engine':<14} │ {'WER':<7} │ {'Avg Lat':<9} │ "
        f"{'P50 Lat':<9} │ {'P95 Lat':<9} │ {'RTF':<7} │ {'Duration':<9} │ {'GPU Mem':<9} │ {'Success':<8} │"
    )
    print(header)
    print("├" + "─" * 108 + "┤")
    
    for rank, r in enumerate(quality_sorted, 1):
        wer_pct = r.avg_roundtrip_wer * 100
        success_str = f"{r.successful_phrases}/{r.total_phrases}"
        print(
            f"│ {rank:<4} │ {r.engine:<14} │ {wer_pct:>5.1f}% │ "
            f"{r.avg_latency:>7.3f}s │ {r.p50_latency:>7.3f}s │ {r.p95_latency:>7.3f}s │ "
            f"{r.avg_rtf:>5.2f}x │ {r.avg_audio_duration:>7.2f}s │ {r.gpu_memory_peak_mb:>7.1f}MB │ {success_str:<8} │"
        )
    
    print("└" + "─" * 108 + "┘")
    
    # Executive Summary Box
    print("\n┌" + "─" * 62 + "┐")
    print("│" + " 📊 EXECUTIVE SUMMARY ".center(62) + "│")
    print("├" + "─" * 62 + "┤")
    
    # Best by quality (WER)
    best_quality = min(valid, key=lambda r: r.avg_roundtrip_wer)
    print(f"│  🏆 Best Quality:      {best_quality.engine:<18} WER: {best_quality.avg_roundtrip_wer*100:>5.1f}%    │")
    
    # Fastest (lowest RTF)  
    fastest = min(valid, key=lambda r: r.avg_rtf)
    print(f"│  ⚡ Fastest (RTF):     {fastest.engine:<18} RTF: {fastest.avg_rtf:>5.2f}x     │")
    
    # Lowest latency
    lowest_lat = min(valid, key=lambda r: r.avg_latency)
    print(f"│  🕐 Lowest Latency:    {lowest_lat.engine:<18} Avg: {lowest_lat.avg_latency:>5.3f}s    │")
    
    # Most memory efficient
    valid_with_mem = [r for r in valid if r.gpu_memory_peak_mb > 0]
    if valid_with_mem:
        lowest_mem = min(valid_with_mem, key=lambda r: r.gpu_memory_peak_mb)
        print(f"│  💾 Lowest GPU Memory: {lowest_mem.engine:<18} Mem: {lowest_mem.gpu_memory_peak_mb:>5.1f}MB   │")
    
    # Best overall (balanced score)
    max_rtf = max(r.avg_rtf for r in valid)
    best_balanced = min(valid, key=lambda r: r.avg_roundtrip_wer + (r.avg_rtf / max_rtf) * 0.5)
    print(f"│  🎯 Best Overall:      {best_balanced.engine:<18} (balanced)     │")
    
    print("├" + "─" * 62 + "┤")
    
    # Statistics summary
    avg_wer_all = statistics.mean(r.avg_roundtrip_wer for r in valid) * 100
    avg_rtf_all = statistics.mean(r.avg_rtf for r in valid)
    avg_duration = statistics.mean(r.avg_audio_duration for r in valid)
    total_success = sum(r.successful_phrases for r in valid)
    total_phrases = sum(r.total_phrases for r in valid)
    
    print(f"│  📈 Average WER across engines:    {avg_wer_all:>5.1f}%                    │")
    print(f"│  📈 Average RTF across engines:    {avg_rtf_all:>5.2f}x                     │")
    print(f"│  🎵 Average audio duration:        {avg_duration:>5.2f}s                     │")
    print(f"│  ✅ Total successful:              {total_success}/{total_phrases} phrases                │")
    print(f"│  🔢 Engines tested:                {len(valid)} passed, {len(failed)} failed           │")
    
    print("└" + "─" * 62 + "┘")
    
    # Performance comparison table (quick reference)
    print("\n┌" + "─" * 50 + "┐")
    print("│" + " 🏁 QUICK COMPARISON ".center(50) + "│")
    print("├" + "─" * 50 + "┤")
    print(f"│ {'Engine':<14} │ {'Quality':<10} │ {'Speed':<10} │ {'Overall':<8} │")
    print("├" + "─" * 50 + "┤")
    
    # Rate each engine
    for r in quality_sorted:
        # Quality rating (based on WER)
        if r.avg_roundtrip_wer < 0.10:
            quality = "★★★★★"
        elif r.avg_roundtrip_wer < 0.20:
            quality = "★★★★☆"
        elif r.avg_roundtrip_wer < 0.35:
            quality = "★★★☆☆"
        elif r.avg_roundtrip_wer < 0.50:
            quality = "★★☆☆☆"
        else:
            quality = "★☆☆☆☆"
        
        # Speed rating (based on RTF, lower is better)
        if r.avg_rtf < 0.3:
            speed = "★★★★★"
        elif r.avg_rtf < 0.7:
            speed = "★★★★☆"
        elif r.avg_rtf < 1.2:
            speed = "★★★☆☆"
        elif r.avg_rtf < 2.0:
            speed = "★★☆☆☆"
        else:
            speed = "★☆☆☆☆"
        
        # Overall rating
        score = (5 - r.avg_roundtrip_wer * 10) * 0.6 + (5 - min(r.avg_rtf, 2) * 2) * 0.4
        if score > 4:
            overall = "🥇"
        elif score > 3:
            overall = "🥈"
        elif score > 2:
            overall = "🥉"
        else:
            overall = "  "
        
        print(f"│ {r.engine:<14} │ {quality:<10} │ {speed:<10} │ {overall:<8} │")
    
    print("└" + "─" * 50 + "┘")
    
    # Print failed engines
    if failed:
        print("\n┌" + "─" * 62 + "┐")
        print("│" + " ❌ FAILED ENGINES ".center(62) + "│")
        print("├" + "─" * 62 + "┤")
        for r in failed:
            error_short = r.error[:44] + "..." if len(r.error) > 44 else r.error
            print(f"│  • {r.engine:<16} {error_short:<40} │")
        print("└" + "─" * 62 + "┘")
    
    # Save JSON results
    json_path = output_dir / "tts_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n📄 Results saved to: {json_path}")
    
    # List audio output directories
    print("\n🔊 Audio samples saved to:")
    for r in valid:
        engine_dir = output_dir / "tts" / r.engine
        if engine_dir.exists():
            num_files = len(list(engine_dir.glob("*.wav")))
            print(f"   └─ {engine_dir} ({num_files} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all TTS engines with latency and quality (roundtrip WER) metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all TTS engines
  python benchmark_tts_engines.py
  
  # Use GPU
  python benchmark_tts_engines.py --use-gpu
  
  # Use a larger judge STT model for more accurate WER
  python benchmark_tts_engines.py --judge-model-size base
  
  # Only benchmark specific engine
  python benchmark_tts_engines.py --only kokoro
        """,
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration for TTS",
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration for TTS (default: auto-detect)",
    )

    parser.add_argument(
        "--judge-model-size",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium"],
        help="Model size for judge STT (default: tiny)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs",
        help="Output directory for results and audio samples (default: benchmark_outputs)",
    )

    parser.add_argument(
        "--only",
        type=str,
        choices=[e.value for e in TTSEngine],
        default=None,
        help="Only benchmark specific TTS engine",
    )
    
    parser.add_argument(
        "--voice-path",
        type=str,
        default=None,
        help="Voice WAV path for Chatterbox (optional)",
    )
    
    parser.add_argument(
        "--vibevoice-speaker",
        type=str,
        default="wayne",
        help="Speaker name for VibeVoice (default: wayne)",
    )

    args = parser.parse_args()
    
    # Determine GPU usage
    if args.no_gpu:
        use_gpu = False
    elif args.use_gpu:
        use_gpu = True
    else:
        use_gpu = check_cuda_availability()
    
    output_dir = Path(args.output_dir)
    tts_output_dir = output_dir / "tts"
    tts_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TTS ENGINE BENCHMARK (with Roundtrip WER)")
    print("=" * 80)
    print(f"\nSystem: {sys.platform}")
    print(f"CUDA Available: {'Yes' if check_cuda_availability() else 'No'}")
    print(f"GPU Mode: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Judge STT Model: {args.judge_model_size}")
    print(f"Output Directory: {output_dir}")
    print(f"Phrases to test: {len(TTS_BENCHMARK_PHRASES)}")

    # Check GPU availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem:.1f} GB")
    elif use_gpu:
        print("\nWarning: GPU requested but not available, using CPU")
        use_gpu = False

    # Initialize GPU monitor
    gpu_monitor = GPUMemoryMonitor()
    
    # Initialize judge STT
    print("\n-> Initializing judge STT (faster-whisper)...")
    try:
        judge_stt = create_stt_engine(
            engine=STTEngine.FASTER_WHISPER,
            device="cuda" if use_gpu else "cpu",
            model_size=args.judge_model_size,
        )
        print(f"   Judge STT ready (model: {args.judge_model_size})")
    except Exception as e:
        print(f"ERROR: Failed to initialize judge STT: {e}")
        print("Cannot compute roundtrip WER without judge STT")
        sys.exit(1)
    
    # Determine which engines to benchmark
    if args.only:
        engines_to_test = [TTSEngine(args.only)]
    else:
        engines_to_test = list(TTSEngine)
    
    print(f"\n-> Engines to benchmark: {[e.value for e in engines_to_test]}")

    # Run benchmarks
    results = []
    
    for engine in engines_to_test:
        result = benchmark_tts_engine(
            engine=engine,
            phrases=TTS_BENCHMARK_PHRASES,
            judge_stt=judge_stt,
            use_gpu=use_gpu,
            output_dir=tts_output_dir,
            gpu_monitor=gpu_monitor,
            voice_path=args.voice_path,
            speaker=args.vibevoice_speaker,
        )
        results.append(result)
        
        # Brief pause between engines
        time.sleep(1)

    # Print results
    print_tts_results_table(results, output_dir)

    print("\n✅ TTS Benchmark complete!")


if __name__ == "__main__":
    main()
