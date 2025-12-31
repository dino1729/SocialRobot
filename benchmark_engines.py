"""Benchmark script comparing different STT and TTS engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import numpy as np
from dotenv import load_dotenv

from audio.engine_config import (
    STTEngine,
    TTSEngine,
    create_stt_engine,
    create_tts_engine,
    word_error_rate,
)
from llm.ollama import OllamaClient
from benchmark import load_audio_file, save_audio_file, check_cuda_availability


# Load environment variables
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")


# =============================================================================
# STT Benchmark Mode - phrase list and utilities
# =============================================================================

STT_BENCHMARK_PHRASES = [
    # Short conversational
    "Hello there! How are you doing today? I hope everything is going well for you.",
    
    # Technical explanation
    "Artificial intelligence works by processing large amounts of data through neural networks. "
    "These networks learn patterns and can make predictions or generate responses based on their training.",
    
    # Complex instructions
    "Please set a timer for fifteen minutes, then remind me to check the oven. "
    "After that, I need you to play some relaxing music in the living room.",
    
    # Narrative paragraph
    "The quick brown fox jumped over the lazy dog while the cat watched from the windowsill. "
    "It was a beautiful autumn afternoon, and the leaves were falling gently from the trees.",
    
    # Question sequence
    "What's the weather going to be like this weekend? Should I bring an umbrella? "
    "Also, can you tell me the best time to visit the farmers market downtown?",
    
    # Numerical and factual
    "The meeting is scheduled for three forty-five on Tuesday, December seventeenth. "
    "We expect approximately twenty-five attendees, and the conference room can hold up to thirty people.",
    
    # Conversational with emotions
    "I'm really excited about the new project we're starting next week! "
    "The team has been working so hard, and I think we're finally ready to launch. "
    "Do you have any suggestions for the presentation?",
    
    # Scientific description
    "The human brain contains approximately eighty-six billion neurons, each forming thousands of connections. "
    "This incredible network enables us to think, remember, and experience consciousness in ways we still don't fully understand.",
    
    # Travel and directions
    "To get to the museum, take the second left after the traffic light, then continue straight for about half a mile. "
    "You'll see the parking garage on your right, just past the coffee shop on the corner.",
    
    # Mixed content with names and places
    "Sarah called this morning to confirm our dinner reservation at Giovanni's Italian Restaurant. "
    "She said the table is booked for seven thirty, and we should try their famous homemade pasta with truffle sauce.",
    
    # Bhagavad Gita - ancient Sanskrit terms (Chapter 2, Verse 47)
    "Karmanye vadhikaraste ma phaleshu kadachana. You have the right to perform your prescribed duties, "
    "but you are not entitled to the fruits of your actions. Never consider yourself the cause of the results, "
    "and never be attached to inaction. This is the essence of Nishkama Karma, selfless action without attachment.",
    
    # Bhagavad Gita - philosophical concept (Chapter 2, Verse 20)
    "The Atman, the eternal soul, is neither born nor does it ever die. It is unborn, eternal, ever-existing, "
    "and primeval. The soul is not slain when the body is slain. This teaching of the immortal Self is central "
    "to the Sankhya philosophy explained by Lord Krishna to Arjuna on the battlefield of Kurukshetra.",
]


@dataclass
class STTBenchmarkResult:
    """Results from benchmarking a single STT engine."""
    engine: str
    init_time: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    avg_rtf: float  # real-time factor
    avg_wer: float
    total_phrases: int
    successful_phrases: int
    error: Optional[str] = None


def generate_stt_dataset(
    phrases: List[str],
    output_dir: Path,
    tts_engine: TTSEngine = TTSEngine.CHATTERBOX,
    voice_paths: Optional[List[str]] = None,
    use_gpu: bool = False,
) -> List[dict]:
    """Generate audio files from phrases using a TTS engine.

    Caches results so repeated runs are fast.

    Args:
        phrases: List of phrases to synthesize
        output_dir: Directory to save audio files
        tts_engine: TTS engine used to generate dataset audio
        voice_paths: Optional list of voice reference WAVs (used by some engines)
        use_gpu: Use GPU for TTS if available

    Returns:
        List of dicts with 'phrase', 'audio_path', 'sample_rate', 'duration', and optional 'voice'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    def dataset_config() -> dict:
        return {
            "tts_engine": tts_engine.value,
            "use_gpu": bool(use_gpu),
            "phrases_sha256": hashlib.sha256("\n".join(phrases).encode("utf-8")).hexdigest(),
            "voice_paths": voice_paths or [],
        }

    # Check if dataset already exists
    if manifest_path.exists():
        print("-> Loading cached STT dataset...")
        with open(manifest_path, "r") as f:
            loaded = json.load(f)

        # Backwards-compatible manifest format (older versions stored a plain list)
        if isinstance(loaded, dict) and "items" in loaded:
            cached_config = loaded.get("config", {})
            manifest_items = loaded.get("items", [])
        else:
            cached_config = {}
            manifest_items = loaded

        # Verify all files exist and config matches
        all_exist = all(Path(item["audio_path"]).exists() for item in manifest_items)
        config_matches = cached_config == dataset_config()
        if all_exist and config_matches:
            print(f"   Found {len(manifest_items)} cached audio files")
            return manifest_items
        print("   Cache incomplete, regenerating...")

    def save_wav(audio_path: Path, audio_data: np.ndarray, sample_rate: int) -> float:
        audio_int16 = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return len(audio_data) / sample_rate

    manifest_items: List[dict] = []

    if tts_engine == TTSEngine.CHATTERBOX:
        voices = voice_paths or []
        if not voices:
            raise ValueError("Chatterbox dataset generation requires at least 1 voice path")

        print(f"-> Generating STT benchmark dataset with Chatterbox TTS ({len(voices)} voices)...")

        for voice_index, voice_path in enumerate(voices):
            voice_label = Path(voice_path).stem
            voice_output_dir = output_dir / voice_label
            voice_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                tts = create_tts_engine(
                    engine=TTSEngine.CHATTERBOX,
                    use_gpu=use_gpu,
                    voice_path=voice_path,
                    use_turbo=True,
                )
            except Exception as e:
                print(f"ERROR: Failed to initialize Chatterbox TTS for voice '{voice_path}': {e}")
                raise

            sample_rate = getattr(tts, "sample_rate", 24000)

            for phrase_index, phrase in enumerate(phrases):
                audio_path = voice_output_dir / f"phrase_{phrase_index:03d}.wav"
                print(
                    f"   [voice {voice_index+1}/{len(voices)} | phrase {phrase_index+1}/{len(phrases)}] "
                    f"Synthesizing: {phrase[:50]}..."
                )

                try:
                    audio_data = tts.synthesize(phrase)
                    if len(audio_data) == 0:
                        print(f"   WARNING: Empty audio for voice '{voice_label}', phrase {phrase_index}")
                        continue

                    duration = save_wav(audio_path, audio_data, sample_rate)
                    manifest_items.append(
                        {
                            "phrase": phrase,
                            "voice": voice_label,
                            "audio_path": str(audio_path),
                            "sample_rate": sample_rate,
                            "duration": duration,
                        }
                    )
                except Exception as e:
                    print(f"   ERROR synthesizing voice '{voice_label}', phrase {phrase_index}: {e}")
                    continue
    else:
        if tts_engine == TTSEngine.KOKORO:
            print("-> Generating STT benchmark dataset with Kokoro TTS...")
            tts_kwargs = {"use_gpu": use_gpu, "voice": "af_bella", "speed": 1.0}
        elif tts_engine == TTSEngine.PIPER:
            print("-> Generating STT benchmark dataset with Piper TTS...")
            tts_kwargs = {"use_gpu": use_gpu}
        elif tts_engine == TTSEngine.VIBEVOICE:
            print("-> Generating STT benchmark dataset with VibeVoice TTS...")
            tts_kwargs = {"use_gpu": use_gpu}
        else:
            raise ValueError(f"Unsupported dataset TTS engine: {tts_engine.value}")

        try:
            tts = create_tts_engine(engine=tts_engine, **tts_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to initialize dataset TTS ({tts_engine.value}): {e}")
            raise

        sample_rate = getattr(tts, "sample_rate", 24000)
        for i, phrase in enumerate(phrases):
            audio_path = output_dir / f"phrase_{i:03d}.wav"
            print(f"   [{i+1}/{len(phrases)}] Synthesizing: {phrase[:50]}...")
            try:
                audio_data = tts.synthesize(phrase)
                if len(audio_data) == 0:
                    print(f"   WARNING: Empty audio for phrase {i}")
                    continue
                duration = save_wav(audio_path, audio_data, sample_rate)
                manifest_items.append(
                    {
                        "phrase": phrase,
                        "audio_path": str(audio_path),
                        "sample_rate": sample_rate,
                        "duration": duration,
                    }
                )
            except Exception as e:
                print(f"   ERROR synthesizing phrase {i}: {e}")
                continue

    # Save manifest (new format includes config for cache validation)
    with open(manifest_path, "w") as f:
        json.dump({"config": dataset_config(), "items": manifest_items}, f, indent=2)

    print(f"-> Generated {len(manifest_items)} audio files in {output_dir}")
    return manifest_items


def benchmark_stt_engine(
    engine: STTEngine,
    dataset: List[dict],
    use_gpu: bool = False,
    num_runs: int = 1,
    model_size: str = "tiny",
    collect_details: bool = True,
) -> Tuple[STTBenchmarkResult, List[dict]]:
    """Benchmark a single STT engine on the dataset.
    
    Args:
        engine: STT engine to benchmark
        dataset: List of audio file info dicts
        use_gpu: Use GPU if available
        num_runs: Number of inference runs per sample (for timing)
        model_size: Model size to use
        
        collect_details: If True, collect per-file transcriptions/metrics
        
    Returns:
        (STTBenchmarkResult, details) where details is a list of per-file dicts
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine.value}")
    print(f"{'='*60}")
    
    # Initialize engine
    print("-> Initializing...")
    try:
        init_start = time.time()
        stt = create_stt_engine(
            engine=engine,
            device="cuda" if use_gpu else "cpu",
            model_size=model_size,
        )
        init_time = time.time() - init_start
        print(f"   Init time: {init_time:.3f}s")
    except Exception as e:
        print(f"ERROR: Failed to initialize {engine.value}: {e}")
        return STTBenchmarkResult(
            engine=engine.value,
            init_time=0.0,
            avg_latency=0.0,
            p50_latency=0.0,
            p95_latency=0.0,
            avg_rtf=0.0,
            avg_wer=1.0,
            total_phrases=len(dataset),
            successful_phrases=0,
            error=str(e),
        ), []
    
    latencies = []
    rtfs = []
    wers = []
    successful = 0
    details: List[dict] = []
    
    for item in dataset:
        phrase = item["phrase"]
        audio_path = item["audio_path"]
        duration = item["duration"]
        sample_rate = item["sample_rate"]
        voice = item.get("voice")
        
        # Load audio
        try:
            raw_bytes, sr, _ = load_audio_file(audio_path)
        except Exception as e:
            print(f"   ERROR loading {audio_path}: {e}")
            continue
        
        # Run inference (multiple times if requested)
        run_latencies = []
        transcription = ""
        
        for run in range(num_runs):
            start = time.time()
            try:
                transcription = stt.run_stt(raw_bytes, sample_rate=sr)
            except Exception as e:
                print(f"   ERROR transcribing {audio_path}: {e}")
                break
            run_latencies.append(time.time() - start)
        
        if not run_latencies:
            continue
        
        # Calculate metrics for this sample
        avg_run_latency = statistics.mean(run_latencies)
        latencies.append(avg_run_latency)
        
        rtf = avg_run_latency / duration if duration > 0 else float("inf")
        rtfs.append(rtf)
        
        wer = word_error_rate(phrase, transcription)
        wers.append(wer)
        
        successful += 1

        if collect_details:
            details.append(
                {
                    "audio_path": audio_path,
                    "voice": voice,
                    "phrase": phrase,
                    "transcription": transcription,
                    "wer": wer,
                    "latency_s": avg_run_latency,
                    "rtf": rtf,
                    "duration_s": duration,
                    "sample_rate": sample_rate,
                    "num_runs": num_runs,
                }
            )
        
        # Show progress
        wer_pct = wer * 100
        print(f"   [{successful}/{len(dataset)}] WER={wer_pct:5.1f}% RTF={rtf:.2f}x | \"{transcription[:40]}...\"")
    
    # Aggregate results
    if not latencies:
        return STTBenchmarkResult(
            engine=engine.value,
            init_time=init_time,
            avg_latency=0.0,
            p50_latency=0.0,
            p95_latency=0.0,
            avg_rtf=0.0,
            avg_wer=1.0,
            total_phrases=len(dataset),
            successful_phrases=0,
            error="No successful transcriptions",
        ), details
    
    sorted_latencies = sorted(latencies)
    p50_idx = int(len(sorted_latencies) * 0.50)
    p95_idx = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)
    
    return STTBenchmarkResult(
        engine=engine.value,
        init_time=init_time,
        avg_latency=statistics.mean(latencies),
        p50_latency=sorted_latencies[p50_idx],
        p95_latency=sorted_latencies[p95_idx],
        avg_rtf=statistics.mean(rtfs),
        avg_wer=statistics.mean(wers),
        total_phrases=len(dataset),
        successful_phrases=successful,
    ), details


def print_stt_results_table(results: List[STTBenchmarkResult], output_dir: Path) -> None:
    """Print and save STT benchmark results table with comprehensive summary."""
    print("\n")
    print("╔" + "═" * 98 + "╗")
    print("║" + " STT ENGINE BENCHMARK RESULTS ".center(98) + "║")
    print("╚" + "═" * 98 + "╝")
    
    # Filter successful results
    valid = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    
    if not valid:
        print("No successful benchmark results.")
        return
    
    # Sort by WER (quality), then by latency
    valid.sort(key=lambda r: (r.avg_wer, r.avg_latency))
    
    # Print detailed results table
    print("\n┌" + "─" * 98 + "┐")
    print("│" + " DETAILED RESULTS (sorted by Word Error Rate) ".center(98) + "│")
    print("├" + "─" * 98 + "┤")
    
    header = (
        f"│ {'Rank':<4} │ {'Engine':<18} │ {'WER':<7} │ {'Avg Lat':<9} │ "
        f"{'P50 Lat':<9} │ {'P95 Lat':<9} │ {'RTF':<7} │ {'Init':<7} │ {'Success':<8} │"
    )
    print(header)
    print("├" + "─" * 98 + "┤")
    
    for rank, r in enumerate(valid, 1):
        wer_pct = r.avg_wer * 100
        success_str = f"{r.successful_phrases}/{r.total_phrases}"
        print(
            f"│ {rank:<4} │ {r.engine:<18} │ {wer_pct:>5.1f}% │ "
            f"{r.avg_latency:>7.3f}s │ {r.p50_latency:>7.3f}s │ {r.p95_latency:>7.3f}s │ "
            f"{r.avg_rtf:>5.2f}x │ {r.init_time:>5.2f}s │ {success_str:<8} │"
        )
    
    print("└" + "─" * 98 + "┘")
    
    # Executive Summary Box
    print("\n┌" + "─" * 58 + "┐")
    print("│" + " 📊 EXECUTIVE SUMMARY ".center(58) + "│")
    print("├" + "─" * 58 + "┤")
    
    # Best by quality
    best_quality = min(valid, key=lambda r: r.avg_wer)
    print(f"│  🏆 Best Quality:    {best_quality.engine:<20} WER: {best_quality.avg_wer*100:>5.1f}%  │")
    
    # Fastest (lowest RTF)
    fastest = min(valid, key=lambda r: r.avg_rtf)
    print(f"│  ⚡ Fastest:         {fastest.engine:<20} RTF: {fastest.avg_rtf:>5.2f}x   │")
    
    # Lowest latency
    lowest_lat = min(valid, key=lambda r: r.avg_latency)
    print(f"│  🕐 Lowest Latency:  {lowest_lat.engine:<20} Avg: {lowest_lat.avg_latency:>5.3f}s  │")
    
    # Best overall (balanced score: WER + normalized RTF)
    # Lower is better for both
    max_rtf = max(r.avg_rtf for r in valid)
    best_balanced = min(valid, key=lambda r: r.avg_wer + (r.avg_rtf / max_rtf) * 0.5)
    print(f"│  🎯 Best Overall:    {best_balanced.engine:<20} (balanced)   │")
    
    print("├" + "─" * 58 + "┤")
    
    # Statistics summary
    avg_wer_all = statistics.mean(r.avg_wer for r in valid) * 100
    avg_rtf_all = statistics.mean(r.avg_rtf for r in valid)
    total_success = sum(r.successful_phrases for r in valid)
    total_phrases = sum(r.total_phrases for r in valid)
    
    print(f"│  📈 Average WER across engines:  {avg_wer_all:>5.1f}%                  │")
    print(f"│  📈 Average RTF across engines:  {avg_rtf_all:>5.2f}x                   │")
    print(f"│  ✅ Total successful:            {total_success}/{total_phrases} phrases              │")
    print(f"│  🔢 Engines tested:              {len(valid)} passed, {len(failed)} failed         │")
    
    print("└" + "─" * 58 + "┘")
    
    # Print failed engines
    if failed:
        print("\n┌" + "─" * 58 + "┐")
        print("│" + " ❌ FAILED ENGINES ".center(58) + "│")
        print("├" + "─" * 58 + "┤")
        for r in failed:
            error_short = r.error[:40] + "..." if len(r.error) > 40 else r.error
            print(f"│  • {r.engine:<18} {error_short:<35} │")
        print("└" + "─" * 58 + "┘")
    
    # Save JSON results
    json_path = output_dir / "stt_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n📄 Results saved to: {json_path}")


def run_stt_benchmark(
    use_gpu: bool = False,
    model_size: str = "tiny",
    num_runs: int = 1,
    output_dir: str = "benchmark_outputs",
    dataset_tts_engine: TTSEngine = TTSEngine.CHATTERBOX,
    dataset_voice_paths: Optional[List[str]] = None,
    dataset_num_voices: int = 5,
) -> None:
    """Run STT benchmark on all available engines.
    
    Args:
        use_gpu: Use GPU if available
        model_size: Model size to use for STT
        num_runs: Number of runs per sample for timing
        output_dir: Output directory for results
    """
    output_path = Path(output_dir)

    # Default to 5 chatterbox voices from repo for accent variety.
    if dataset_tts_engine == TTSEngine.CHATTERBOX and not dataset_voice_paths:
        preferred = [
            "voices/jamaican.wav",
            "voices/aave_female.wav",
            "voices/morgan_freeman.wav",
            "voices/snoop_dogg.wav",
            "voices/jerry_seinfeld.wav",
        ]
        dataset_voice_paths = [p for p in preferred if Path(p).exists()]
        if len(dataset_voice_paths) < dataset_num_voices:
            discovered = sorted(str(p) for p in Path("voices").glob("*.wav"))
            for p in discovered:
                if p not in dataset_voice_paths:
                    dataset_voice_paths.append(p)
                if len(dataset_voice_paths) >= dataset_num_voices:
                    break
        dataset_voice_paths = dataset_voice_paths[: max(1, dataset_num_voices)]

    # Separate cache folder per dataset configuration
    voices_key = ",".join(dataset_voice_paths or [])
    effective_use_gpu = use_gpu if dataset_tts_engine != TTSEngine.CHATTERBOX else False
    dataset_hash = hashlib.sha256(
        f"{dataset_tts_engine.value}|{int(effective_use_gpu)}|{voices_key}|{len(STT_BENCHMARK_PHRASES)}".encode("utf-8")
    ).hexdigest()[:10]
    dataset_path = output_path / f"stt_dataset_{dataset_tts_engine.value}_{dataset_hash}"
    
    print("=" * 70)
    print("=== STT ENGINE BENCHMARK ===")
    print("=" * 70)
    print(f"\nSystem: {sys.platform}")
    print(f"CUDA Available: {'Yes' if check_cuda_availability() else 'No'}")
    print(f"GPU Mode: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Model Size: {model_size}")
    print(f"Runs per sample: {num_runs}")
    print(f"Output Directory: {output_path}")
    print(f"Dataset TTS Engine: {dataset_tts_engine.value}")
    if dataset_voice_paths:
        print(f"Dataset Voices: {len(dataset_voice_paths)}")
    
    # Generate dataset
    dataset = generate_stt_dataset(
        phrases=STT_BENCHMARK_PHRASES,
        output_dir=dataset_path,
        tts_engine=dataset_tts_engine,
        voice_paths=dataset_voice_paths,
        use_gpu=use_gpu,
    )
    
    if not dataset:
        print("ERROR: Failed to generate dataset")
        return
    
    # Benchmark each engine
    results = []
    per_engine_details: dict[str, List[dict]] = {}
    for engine in STTEngine:
        result, details = benchmark_stt_engine(
            engine=engine,
            dataset=dataset,
            use_gpu=use_gpu,
            num_runs=num_runs,
            model_size=model_size,
            collect_details=True,
        )
        results.append(result)
        per_engine_details[engine.value] = details

    # Save per-file transcriptions + metrics
    details_path = output_path / "stt_benchmark_transcriptions.json"
    details_payload: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "dataset_tts_engine": dataset_tts_engine.value,
        "dataset_voices": dataset_voice_paths or [],
        "model_size": model_size,
        "use_gpu": bool(use_gpu),
        "num_runs": int(num_runs),
        "generated_at_unix": int(time.time()),
        "per_engine": per_engine_details,
    }
    with open(details_path, "w") as f:
        json.dump(details_payload, f, indent=2)
    print(f"\n📄 Transcriptions saved to: {details_path}")

    # Save a Markdown report with clickable audio links in most editors
    md_path = output_path / "stt_benchmark_transcriptions.md"
    md_lines: List[str] = []
    md_lines.append("# STT Benchmark Transcriptions\n")
    md_lines.append(f"- Dataset: `{dataset_path}`\n")
    md_lines.append(f"- Dataset TTS engine: `{dataset_tts_engine.value}`\n")
    md_lines.append(f"- Voices: {len(dataset_voice_paths or [])}\n")
    md_lines.append(f"- STT model size: `{model_size}`\n")
    md_lines.append(f"- STT GPU: `{use_gpu}`\n")
    md_lines.append("\n")

    for engine_name, samples in per_engine_details.items():
        md_lines.append(f"## {engine_name}\n")
        md_lines.append("| voice | phrase | transcription | WER | audio |\n")
        md_lines.append("|---|---|---|---:|---|\n")
        for s in samples:
            voice = cast(Optional[str], s.get("voice")) or ""
            phrase = cast(str, s.get("phrase", ""))
            transcription = cast(str, s.get("transcription", ""))
            wer = cast(float, s.get("wer", 0.0)) * 100.0
            audio_path = cast(str, s.get("audio_path", ""))
            md_lines.append(
                f"| {voice} | {phrase} | {transcription} | {wer:.1f}% | [{Path(audio_path).name}]({audio_path}) |\n"
            )
        md_lines.append("\n")

    with open(md_path, "w") as f:
        f.writelines(md_lines)
    print(f"📄 Markdown report saved to: {md_path}")
    
    # Print results
    print_stt_results_table(results, output_path)
    
    print("\n✅ STT Benchmark complete!")


def benchmark_engine_combination(
    audio_file: str,
    stt_engine: STTEngine,
    tts_engine: TTSEngine,
    use_gpu: bool = True,
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
    save_audio: bool = True,
    output_dir: str = "benchmark_outputs",
) -> dict:
    """Benchmark a specific combination of STT and TTS engines.
    
    Args:
        audio_file: Path to input audio file
        stt_engine: STT engine to use
        tts_engine: TTS engine to use
        use_gpu: Enable GPU acceleration where available
        ollama_url: Ollama server URL
        ollama_model: Ollama model name
        
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 70)
    print(f"Testing: {stt_engine.value} (STT) + {tts_engine.value} (TTS)")
    print("=" * 70)
    
    # Load audio file
    try:
        raw_bytes, sample_rate, duration = load_audio_file(audio_file)
    except ValueError as e:
        print(f"ERROR: {e}")
        return {}
    
    results = {
        'stt_engine': stt_engine.value,
        'tts_engine': tts_engine.value,
        'stt_init_time': 0.0,
        'stt_inference_time': 0.0,
        'llm_init_time': 0.0,
        'llm_inference_time': 0.0,
        'tts_init_time': 0.0,
        'tts_inference_time': 0.0,
        'transcribed_text': '',
        'llm_response': '',
        'error': None,
    }
    
    try:
        # ====================================================================
        # 1. STT Benchmark
        # ====================================================================
        print("\n-> Initializing STT engine...")
        start_time = time.time()
        
        stt_model = create_stt_engine(
            engine=stt_engine,
            device="cuda" if use_gpu else "cpu",
            model_size="tiny",
        )
        
        results['stt_init_time'] = time.time() - start_time
        print(f"   Init time: {results['stt_init_time']:.3f}s")
        
        print("-> Running STT inference...")
        start_time = time.time()
        recognized_text = stt_model.run_stt(raw_bytes, sample_rate=sample_rate)
        results['stt_inference_time'] = time.time() - start_time
        results['transcribed_text'] = recognized_text
        print(f"   Inference time: {results['stt_inference_time']:.3f}s")
        print(f"   Transcribed: \"{recognized_text[:100]}...\"")
        
        if not recognized_text.strip():
            print("WARNING: No text transcribed")
            results['error'] = "No text transcribed"
            return results
        
        # ====================================================================
        # 2. LLM Benchmark
        # ====================================================================
        print("\n-> Initializing LLM...")
        start_time = time.time()
        
        # Use longer response for large audio benchmarks
        audio_duration = len(raw_bytes) / sample_rate
        if audio_duration > 10:
            system_prompt = "You are a knowledgeable AI assistant. Provide detailed, comprehensive answers with 5-6 sentences, explaining concepts thoroughly with examples and context."
        else:
            system_prompt = "You are a helpful AI assistant. Keep responses brief and concise (1-2 sentences)."
        
        ollama_client = OllamaClient(
            url=ollama_url,
            model=ollama_model,
            stream=True,
            system_prompt=system_prompt,
        )
        results['llm_init_time'] = time.time() - start_time
        print(f"   Init time: {results['llm_init_time']:.3f}s")
        
        print("-> Querying LLM...")
        start_time = time.time()
        llm_response = ollama_client.query(recognized_text)
        results['llm_inference_time'] = time.time() - start_time
        results['llm_response'] = llm_response
        print(f"   Inference time: {results['llm_inference_time']:.3f}s")
        print(f"   Response: \"{llm_response[:100]}...\"")
        
        if not llm_response.strip():
            print("WARNING: No LLM response")
            results['error'] = "No LLM response"
            return results
        
        # ====================================================================
        # 3. TTS Benchmark
        # ====================================================================
        print("\n-> Initializing TTS engine...")
        start_time = time.time()
        
        tts_model = create_tts_engine(
            engine=tts_engine,
            use_gpu=use_gpu,
        )
        
        results['tts_init_time'] = time.time() - start_time
        print(f"   Init time: {results['tts_init_time']:.3f}s")
        
        print("-> Synthesizing speech...")
        start_time = time.time()
        audio_data = tts_model.synthesize(llm_response)
        results['tts_inference_time'] = time.time() - start_time
        print(f"   Inference time: {results['tts_inference_time']:.3f}s")
        print(f"   Generated audio duration: {len(audio_data) / getattr(tts_model, 'sample_rate', 22050):.2f}s")
        
        # Save audio output
        if save_audio and len(audio_data) > 0:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with engine names
            output_filename = f"{stt_engine.value}_{tts_engine.value}_{'gpu' if use_gpu else 'cpu'}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            from benchmark import save_audio_file
            save_audio_file(audio_data, getattr(tts_model, 'sample_rate', 22050), output_path)
            results['audio_output_file'] = output_path
            print(f"   ✓ Saved audio to: {output_path}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        results['error'] = "Interrupted"
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        results['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return results


def print_comparison_table(all_results: list[dict], save_markdown: bool = True, output_dir: str = "benchmark_outputs") -> None:
    """Print comparison table for all engine combinations."""
    print("\n" + "=" * 120)
    print("=== ENGINE COMPARISON SUMMARY ===")
    print("=" * 120)
    
    # Build markdown report
    md_lines = []
    md_lines.append("# Voice Assistant Engine Benchmark Report\n")
    md_lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append("---\n")
    
    # Console output
    header = f"{'Engine Combination':<35} | {'STT Init':<10} | {'STT Infer':<10} | {'LLM Infer':<10} | {'TTS Init':<10} | {'TTS Infer':<10} | {'Total':<10} | {'Status':<10}"
    print("\n" + header)
    print("-" * 120)
    
    # Markdown table header
    md_lines.append("\n## Performance Comparison\n")
    md_lines.append("| Engine Combination | STT Init | STT Infer | LLM Infer | TTS Init | TTS Infer | Total | Status | Audio Output |")
    md_lines.append("|-------------------|----------|-----------|-----------|----------|-----------|-------|--------|--------------|")
    
    for result in all_results:
        # Skip empty results
        if not result or 'stt_engine' not in result:
            continue
            
        if result.get('error'):
            status = "FAILED"
            total = 0.0
        else:
            status = "OK"
            total = (
                result.get('stt_init_time', 0) + result.get('stt_inference_time', 0) +
                result.get('llm_init_time', 0) + result.get('llm_inference_time', 0) +
                result.get('tts_init_time', 0) + result.get('tts_inference_time', 0)
            )
        
        engine_combo = f"{result['stt_engine']} + {result['tts_engine']}"
        audio_file = result.get('audio_output_file', 'N/A')
        
        # Console output
        print(
            f"{engine_combo:<35} | "
            f"{result.get('stt_init_time', 0):>8.3f}s | "
            f"{result.get('stt_inference_time', 0):>8.3f}s | "
            f"{result.get('llm_inference_time', 0):>8.3f}s | "
            f"{result.get('tts_init_time', 0):>8.3f}s | "
            f"{result.get('tts_inference_time', 0):>8.3f}s | "
            f"{total:>8.3f}s | "
            f"{status:<10}"
        )
        
        # Markdown output
        audio_link = f"[🔊 Audio]({audio_file})" if audio_file != 'N/A' else 'N/A'
        md_lines.append(
            f"| {engine_combo} | "
            f"{result.get('stt_init_time', 0):.3f}s | "
            f"{result.get('stt_inference_time', 0):.3f}s | "
            f"{result.get('llm_inference_time', 0):.3f}s | "
            f"{result.get('tts_init_time', 0):.3f}s | "
            f"{result.get('tts_inference_time', 0):.3f}s | "
            f"{total:.3f}s | "
            f"{status} | {audio_link} |"
        )
    
    print("-" * 120)
    
    # Find fastest combination
    valid_results = [r for r in all_results if not r.get('error')]
    if valid_results:
        fastest = min(valid_results, key=lambda r: (
            r.get('stt_init_time', 0) + r.get('stt_inference_time', 0) +
            r.get('llm_init_time', 0) + r.get('llm_inference_time', 0) +
            r.get('tts_init_time', 0) + r.get('tts_inference_time', 0)
        ))
        
        fastest_combo = f"{fastest['stt_engine']} + {fastest['tts_engine']}"
        fastest_time = (
            fastest.get('stt_init_time', 0) + fastest.get('stt_inference_time', 0) +
            fastest.get('llm_init_time', 0) + fastest.get('llm_inference_time', 0) +
            fastest.get('tts_init_time', 0) + fastest.get('tts_inference_time', 0)
        )
        
        print(f"\n🏆 Fastest Combination: {fastest_combo} ({fastest_time:.3f}s total)")
        
        # Add to markdown
        md_lines.append(f"\n## 🏆 Fastest Combination\n")
        md_lines.append(f"**{fastest_combo}** - {fastest_time:.3f}s total\n")
        
        # Add details
        md_lines.append("\n## Detailed Results\n")
        for result in valid_results:
            if not result.get('error'):
                combo = f"{result['stt_engine']} + {result['tts_engine']}"
                md_lines.append(f"\n### {combo}\n")
                md_lines.append(f"- **Transcribed Text:** {result.get('transcribed_text', 'N/A')[:200]}...\n")
                md_lines.append(f"- **LLM Response:** {result.get('llm_response', 'N/A')[:200]}...\n")
                if result.get('audio_output_file'):
                    md_lines.append(f"- **Audio Output:** [{result['audio_output_file']}]({result['audio_output_file']})\n")
    
    print("\n" + "=" * 120)
    
    # Save markdown report
    if save_markdown and md_lines:
        import os
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "benchmark_report.md")
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(md_lines))
        
        print(f"\n📄 Markdown report saved to: {report_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark different STT and TTS engine combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # STT-only benchmark (evaluates all STT engines on generated dataset)
  python benchmark_engines.py --mode stt
  python benchmark_engines.py --mode stt --model-size base

  # Use Chatterbox TTS voices to generate an accent-diverse STT dataset
  python benchmark_engines.py --mode stt --dataset-tts-engine chatterbox --dataset-num-voices 5
  python benchmark_engines.py --mode stt --dataset-tts-engine chatterbox --dataset-voice-paths voices/jamaican.wav voices/aave_female.wav voices/morgan_freeman.wav voices/snoop_dogg.wav voices/jerry_seinfeld.wav
  
  # Pipeline benchmark (STT + LLM + TTS)
  python benchmark_engines.py --mode pipeline --audio-file test.wav --test-all
  
  # Test specific STT engine in pipeline mode
  python benchmark_engines.py --mode pipeline --audio-file test.wav --stt-engine openai-whisper
  
  # Test specific TTS engine in pipeline mode
  python benchmark_engines.py --mode pipeline --audio-file test.wav --tts-engine piper
  
  # Test specific combination
  python benchmark_engines.py --mode pipeline --audio-file test.wav --stt-engine faster-whisper --tts-engine kokoro
  
  # Disable GPU
  python benchmark_engines.py --mode stt --no-gpu
        """,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pipeline", "stt"],
        default="pipeline",
        help="Benchmark mode: 'stt' for STT-only with WER scoring, 'pipeline' for full STT+LLM+TTS (default: pipeline)",
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        required=False,
        help="Path to input WAV audio file (required for pipeline mode)",
    )
    
    parser.add_argument(
        "--stt-engine",
        type=str,
        choices=[e.value for e in STTEngine],
        default=None,
        help="STT engine to test (default: test all)",
    )
    
    parser.add_argument(
        "--tts-engine",
        type=str,
        choices=[e.value for e in TTSEngine],
        default=None,
        help="TTS engine to test (default: test all)",
    )
    
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all engine combinations (pipeline mode)",
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size for STT engines (default: tiny)",
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of inference runs per sample for timing (stt mode, default: 1)",
    )

    parser.add_argument(
        "--dataset-tts-engine",
        type=str,
        choices=[e.value for e in TTSEngine],
        default=TTSEngine.CHATTERBOX.value,
        help="TTS engine used to generate the STT benchmark dataset (stt mode, default: chatterbox)",
    )

    parser.add_argument(
        "--dataset-voice-paths",
        type=str,
        nargs="+",
        default=None,
        help="Voice reference WAV paths for dataset generation (stt mode; used by chatterbox)",
    )

    parser.add_argument(
        "--dataset-num-voices",
        type=int,
        default=5,
        help="Number of voices to auto-select from `voices/` when generating a chatterbox dataset (stt mode, default: 5)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs",
        help="Output directory for results (default: benchmark_outputs)",
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=OLLAMA_URL,
        help=f"Ollama server URL (default: {OLLAMA_URL})",
    )
    
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=OLLAMA_MODEL,
        help=f"Ollama model name (default: {OLLAMA_MODEL})",
    )
    
    args = parser.parse_args()
    
    use_gpu = not args.no_gpu
    
    # Handle STT-only mode
    if args.mode == "stt":
        run_stt_benchmark(
            use_gpu=use_gpu,
            model_size=args.model_size,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            dataset_tts_engine=TTSEngine(args.dataset_tts_engine),
            dataset_voice_paths=args.dataset_voice_paths,
            dataset_num_voices=args.dataset_num_voices,
        )
        return
    
    # Pipeline mode requires audio file
    if not args.audio_file:
        parser.error("--audio-file is required for pipeline mode")
    
    print("=" * 70)
    print("=== VOICE ASSISTANT ENGINE BENCHMARK ===")
    print("=" * 70)
    print(f"\nSystem: {sys.platform}")
    print(f"CUDA Available: {'Yes' if check_cuda_availability() else 'No'}")
    print(f"GPU Mode: {'Disabled' if args.no_gpu else 'Enabled'}")
    print(f"Audio File: {args.audio_file}")
    
    # Determine which combinations to test
    if args.test_all:
        stt_engines = list(STTEngine)
        tts_engines = list(TTSEngine)
    else:
        stt_engines = [STTEngine(args.stt_engine)] if args.stt_engine else list(STTEngine)
        tts_engines = [TTSEngine(args.tts_engine)] if args.tts_engine else list(TTSEngine)
    
    # Run benchmarks
    all_results = []
    
    for stt_engine in stt_engines:
        for tts_engine in tts_engines:
            print("\n")
            result = benchmark_engine_combination(
                audio_file=args.audio_file,
                stt_engine=stt_engine,
                tts_engine=tts_engine,
                use_gpu=use_gpu,
                ollama_url=args.ollama_url,
                ollama_model=args.ollama_model,
                save_audio=True,
                output_dir=args.output_dir,
            )
            all_results.append(result)
            print("\n")
            time.sleep(1)  # Brief pause between tests
    
    # Print comparison table
    if len(all_results) > 1:
        print_comparison_table(all_results, output_dir=args.output_dir)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
