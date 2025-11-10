"""Benchmarking script for voice assistant pipeline (STT → LLM → TTS)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from audio.stt import FasterWhisperSTT
from audio.tts import KokoroTTS
from llm.ollama import OllamaClient


# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")


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


def _detect_onnx_device() -> str:
    """Detects available ONNX runtime execution providers."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            return "CUDA (ONNX)"
        elif 'TensorrtExecutionProvider' in providers:
            return "TensorRT (ONNX)"
        else:
            return "CPU (ONNX)"
    except Exception:
        return "CPU (ONNX)"


def load_audio_file(audio_path: str) -> Tuple[bytes, int, float]:
    """Load a WAV audio file and return raw bytes with metadata.
    
    Args:
        audio_path: Path to the WAV audio file
    
    Returns:
        Tuple of (raw_bytes, sample_rate, duration_seconds)
    
    Raises:
        ValueError: If the file format is invalid or unsupported
    """
    if not os.path.exists(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")
    
    try:
        with wave.open(audio_path, 'rb') as wf:
            # Get audio parameters
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Calculate duration
            duration = n_frames / float(sample_rate)
            
            # Read audio data
            raw_data = wf.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio_np = np.frombuffer(raw_data, dtype=np.int16)
            elif sample_width == 4:  # 32-bit
                audio_np = np.frombuffer(raw_data, dtype=np.int32)
                # Convert to 16-bit
                audio_np = (audio_np / 65536).astype(np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width} bytes")
            
            # Convert to mono if stereo
            if n_channels == 2:
                audio_np = audio_np.reshape(-1, 2).mean(axis=1).astype(np.int16)
            elif n_channels != 1:
                raise ValueError(f"Unsupported number of channels: {n_channels}")
            
            # Convert back to bytes
            raw_bytes = audio_np.tobytes()
            
            return raw_bytes, sample_rate, duration
            
    except wave.Error as e:
        raise ValueError(f"Invalid WAV file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")


def save_audio_file(audio_data: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Save audio data to a WAV file.
    
    Args:
        audio_data: Audio data as float32 numpy array
        sample_rate: Sample rate in Hz
        output_path: Path to save the WAV file
    """
    # Convert float32 to int16
    audio_int16 = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
    
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"-> Saved generated audio to: {output_path}")


def check_cuda_availability() -> bool:
    """Check if CUDA is available."""
    try:
        import ctranslate2  # type: ignore
        return ctranslate2.get_cuda_device_count() > 0  # type: ignore[attr-defined]
    except Exception:
        return False


def _run_single_benchmark(
    raw_bytes: bytes,
    sample_rate: int,
    duration: float,
    channels: str,
    audio_file: str,
    compute_type: Optional[str],
    ollama_url: str,
    ollama_model: str,
    output_audio: Optional[str],
    report_file: Optional[str],
    stt_device: str,
    tts_use_gpu: bool,
    label: str = "",
) -> dict:
    """Run a single benchmark configuration.
    
    Returns:
        dict: Benchmark results
    """
    # Determine compute type
    if compute_type is None:
        compute_type = _get_default_compute_type(stt_device)
        print(f"\n-> Auto-detected compute type: {compute_type} (device: {stt_device})")
    else:
        print(f"\n-> Using specified compute type: {compute_type} (device: {stt_device})")
    
    # Initialize timing results
    results = {
        'stt_init_time': 0.0,
        'stt_inference_time': 0.0,
        'stt_device': stt_device.upper(),
        'stt_compute_type': compute_type,
        'llm_init_time': 0.0,
        'llm_inference_time': 0.0,
        'tts_init_time': 0.0,
        'tts_inference_time': 0.0,
        'tts_device': '',
        'tts_gpu_enabled': tts_use_gpu,
        'transcribed_text': '',
        'llm_response': '',
    }
    
    # ========================================================================
    # 1. STT Benchmark
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. Speech-to-Text (Faster-Whisper) Benchmark")
    print("=" * 70)
    
    print(f"-> Initializing FasterWhisperSTT (model='tiny.en', device={stt_device}, compute_type={compute_type})...")
    start_time = time.time()
    stt_model = FasterWhisperSTT(
        model_size_or_path="tiny.en",
        device=stt_device,
        compute_type=compute_type
    )
    results['stt_init_time'] = time.time() - start_time
    print(f"   Initialization time: {results['stt_init_time']:.3f}s")
    
    print(f"-> Running STT inference on {duration:.2f}s audio...")
    start_time = time.time()
    recognized_text = stt_model.run_stt(raw_bytes, sample_rate=sample_rate)
    results['stt_inference_time'] = time.time() - start_time
    results['transcribed_text'] = recognized_text
    print(f"   Inference time: {results['stt_inference_time']:.3f}s")
    print(f"   Transcribed: \"{recognized_text}\"")
    
    if not recognized_text.strip():
        print("\nWARNING: No text was transcribed. Cannot proceed with LLM and TTS benchmarks.")
        return results
    
    # ========================================================================
    # 2. LLM Benchmark
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. Large Language Model (Ollama) Benchmark")
    print("=" * 70)
    
    print(f"-> Initializing OllamaClient (url={ollama_url}, model={ollama_model})...")
    start_time = time.time()
    
    # Use longer response for large audio benchmarks
    audio_duration = len(raw_bytes) / sample_rate
    if audio_duration > 10:
        system_prompt = "You are a world-class knowledgeable AI voice assistant. Provide detailed, comprehensive answers with 5-6 sentences, explaining concepts thoroughly with examples and context."
    else:
        system_prompt = "You are a world-class knowledgeable AI voice assistant, Orion. Your mission is to assist users with any questions or tasks they have. Keep answers brief, concise and within 1-2 sentences. Be polite, friendly, and respectful."
    
    ollama_client = OllamaClient(
        url=ollama_url,
        model=ollama_model,
        stream=True,
        system_prompt=system_prompt,
    )
    results['llm_init_time'] = time.time() - start_time
    print(f"   Initialization time: {results['llm_init_time']:.3f}s")
    
    print(f"-> Querying Ollama with transcribed text...")
    start_time = time.time()
    llm_response = ollama_client.query(recognized_text)
    results['llm_inference_time'] = time.time() - start_time
    results['llm_response'] = llm_response
    print(f"   Inference time: {results['llm_inference_time']:.3f}s")
    if results['llm_inference_time'] > 5.0:
        print(f"   Note: Inference time >5s likely includes Ollama model loading")
    print(f"   Response: \"{llm_response}\"")
    
    if not llm_response.strip():
        print("\nWARNING: No response from LLM. Cannot proceed with TTS benchmark.")
        return results
    
    # ========================================================================
    # 3. TTS Benchmark
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. Text-to-Speech (Kokoro-ONNX) Benchmark")
    print("=" * 70)
    
    print(f"-> Initializing KokoroTTS (voice='af_bella', speed=1.0, use_gpu={tts_use_gpu})...")
    start_time = time.time()
    tts_model = KokoroTTS(voice="af_bella", speed=1.0, use_gpu=tts_use_gpu)
    results['tts_init_time'] = time.time() - start_time
    print(f"   Initialization time: {results['tts_init_time']:.3f}s")
    
    # Detect actual TTS device
    results['tts_device'] = tts_model._get_actual_provider()
    
    print(f"-> Synthesizing speech from LLM response...")
    start_time = time.time()
    audio_data = tts_model.synthesize(llm_response)
    results['tts_inference_time'] = time.time() - start_time
    print(f"   Inference time: {results['tts_inference_time']:.3f}s")
    print(f"   Generated audio duration: {len(audio_data) / tts_model.sample_rate:.2f}s")
    
    # Save output audio if requested (only for non-comparison runs)
    if output_audio and not label:
        save_audio_file(audio_data, tts_model.sample_rate, output_audio)
    
    # Generate report
    _print_report(results, duration, sample_rate, channels, audio_file, ollama_model, label)
    
    # Save report to file if requested (only for non-comparison runs)
    if report_file and not label:
        _save_report(results, duration, sample_rate, channels, audio_file, ollama_model, report_file)
    
    return results


def _print_report(
    results: dict,
    duration: float,
    sample_rate: int,
    channels: str,
    audio_file: str,
    ollama_model: str,
    label: str = "",
) -> None:
    """Print benchmark report to console."""
    print("\n" + "=" * 70)
    print(f"=== BENCHMARK REPORT {('- ' + label) if label else ''} ===")
    print("=" * 70)
    
    cuda_available = check_cuda_availability()
    print(f"\nAudio File: {Path(audio_file).name} ({duration:.2f}s, {sample_rate}Hz, {channels})")
    print(f"System: {sys.platform}, CUDA Available: {'Yes' if cuda_available else 'No'}")
    print(f"Ollama Model: {ollama_model}")
    
    # Calculate total time
    total_time = (
        results['stt_init_time'] + results['stt_inference_time'] +
        results['llm_init_time'] + results['llm_inference_time'] +
        results['tts_init_time'] + results['tts_inference_time']
    )
    
    # Print table
    print("\n" + "-" * 100)
    print(f"{'Step':<20} | {'Init Time':<12} | {'Inference Time':<15} | {'Device':<15} | {'Compute Type':<15}")
    print("-" * 100)
    
    llm_note = "*" if results['llm_inference_time'] > 5.0 else ""
    print(f"{'STT (Whisper)':<20} | {results['stt_init_time']:>10.3f}s | {results['stt_inference_time']:>13.3f}s | {results['stt_device']:<15} | {results['stt_compute_type']:<15}")
    print(f"{'LLM (Ollama)':<20} | {results['llm_init_time']:>10.3f}s | {results['llm_inference_time']:>13.3f}s{llm_note} | {'GPU (Ollama)':<15} | {'N/A':<15}")
    print(f"{'TTS (Kokoro)':<20} | {results['tts_init_time']:>10.3f}s | {results['tts_inference_time']:>13.3f}s | {results['tts_device']:<15} | {'N/A':<15}")
    print("-" * 100)
    print(f"{'TOTAL PIPELINE':<20} | {'':<12} | {total_time:>13.3f}s | {'':<15} | {'':<15}")
    print("-" * 100)
    
    if results['llm_inference_time'] > 5.0:
        print(f"\n* Note: First Ollama call includes model loading time (~{results['llm_inference_time']:.1f}s)")
    
    print(f"\nTranscribed Text: {results['transcribed_text']}")
    print(f"Generated Response: {results['llm_response']}")
    
    print("\n" + "=" * 70)
    print("=== Benchmark Complete ===")
    print("=" * 70)


def _print_comparison_summary(cpu_results: dict, gpu_results: dict) -> None:
    """Print a comparison summary showing speed-up between CPU and GPU."""
    print("\n" + "=" * 70)
    print("=== COMPARISON SUMMARY ===")
    print("=" * 70)
    
    # Calculate total times
    cpu_total = (
        cpu_results['stt_init_time'] + cpu_results['stt_inference_time'] +
        cpu_results['llm_init_time'] + cpu_results['llm_inference_time'] +
        cpu_results['tts_init_time'] + cpu_results['tts_inference_time']
    )
    
    gpu_total = (
        gpu_results['stt_init_time'] + gpu_results['stt_inference_time'] +
        gpu_results['llm_init_time'] + gpu_results['llm_inference_time'] +
        gpu_results['tts_init_time'] + gpu_results['tts_inference_time']
    )
    
    # Calculate speed-up
    speedup = cpu_total / gpu_total if gpu_total > 0 else 0
    time_saved = cpu_total - gpu_total
    percent_faster = ((cpu_total - gpu_total) / cpu_total * 100) if cpu_total > 0 else 0
    
    print("\n" + "-" * 70)
    print(f"{'Metric':<30} | {'CPU Mode':<15} | {'GPU Mode':<15}")
    print("-" * 70)
    
    # STT times
    cpu_stt = cpu_results['stt_inference_time']
    gpu_stt = gpu_results['stt_inference_time']
    stt_speedup = cpu_stt / gpu_stt if gpu_stt > 0 else 0
    print(f"{'STT Inference Time':<30} | {cpu_stt:>13.3f}s | {gpu_stt:>13.3f}s  ({stt_speedup:.2f}x)")
    
    # LLM times (both use GPU via Ollama)
    cpu_llm = cpu_results['llm_inference_time']
    gpu_llm = gpu_results['llm_inference_time']
    print(f"{'LLM Inference Time':<30} | {cpu_llm:>13.3f}s | {gpu_llm:>13.3f}s")
    
    # TTS times
    cpu_tts = cpu_results['tts_inference_time']
    gpu_tts = gpu_results['tts_inference_time']
    tts_speedup = cpu_tts / gpu_tts if gpu_tts > 0 and gpu_results['tts_gpu_enabled'] else 1.0
    tts_note = "" if gpu_results['tts_gpu_enabled'] and 'GPU' in gpu_results['tts_device'] else "  (CPU fallback)"
    print(f"{'TTS Inference Time':<30} | {cpu_tts:>13.3f}s | {gpu_tts:>13.3f}s{tts_note}")
    
    print("-" * 70)
    print(f"{'TOTAL PIPELINE TIME':<30} | {cpu_total:>13.3f}s | {gpu_total:>13.3f}s")
    print("-" * 70)
    
    print(f"\n🚀 GPU Speed-up: {speedup:.2f}x faster ({percent_faster:.1f}% improvement)")
    print(f"⏱️  Time Saved: {time_saved:.3f}s")
    
    print("\n" + "=" * 70)


def _save_report(
    results: dict,
    duration: float,
    sample_rate: int,
    channels: str,
    audio_file: str,
    ollama_model: str,
    report_file: str,
) -> None:
    """Save benchmark report to file."""
    report_lines = []
    
    report_lines.append("=" * 70)
    report_lines.append("=== BENCHMARK REPORT ===")
    report_lines.append("=" * 70)
    
    cuda_available = check_cuda_availability()
    report_lines.append(f"\nAudio File: {Path(audio_file).name} ({duration:.2f}s, {sample_rate}Hz, {channels})")
    report_lines.append(f"System: {sys.platform}, CUDA Available: {'Yes' if cuda_available else 'No'}")
    report_lines.append(f"Ollama Model: {ollama_model}")
    
    # Calculate total time
    total_time = (
        results['stt_init_time'] + results['stt_inference_time'] +
        results['llm_init_time'] + results['llm_inference_time'] +
        results['tts_init_time'] + results['tts_inference_time']
    )
    
    # Print table
    report_lines.append("\n" + "-" * 100)
    report_lines.append(f"{'Step':<20} | {'Init Time':<12} | {'Inference Time':<15} | {'Device':<15} | {'Compute Type':<15}")
    report_lines.append("-" * 100)
    
    llm_note = "*" if results['llm_inference_time'] > 5.0 else ""
    report_lines.append(f"{'STT (Whisper)':<20} | {results['stt_init_time']:>10.3f}s | {results['stt_inference_time']:>13.3f}s | {results['stt_device']:<15} | {results['stt_compute_type']:<15}")
    report_lines.append(f"{'LLM (Ollama)':<20} | {results['llm_init_time']:>10.3f}s | {results['llm_inference_time']:>13.3f}s{llm_note} | {'GPU (Ollama)':<15} | {'N/A':<15}")
    report_lines.append(f"{'TTS (Kokoro)':<20} | {results['tts_init_time']:>10.3f}s | {results['tts_inference_time']:>13.3f}s | {results['tts_device']:<15} | {'N/A':<15}")
    report_lines.append("-" * 100)
    report_lines.append(f"{'TOTAL PIPELINE':<20} | {'':<12} | {total_time:>13.3f}s | {'':<15} | {'':<15}")
    report_lines.append("-" * 100)
    
    if results['llm_inference_time'] > 5.0:
        report_lines.append(f"\n* Note: First Ollama call includes model loading time (~{results['llm_inference_time']:.1f}s)")
    
    report_lines.append(f"\nTranscribed Text: {results['transcribed_text']}")
    report_lines.append(f"Generated Response: {results['llm_response']}")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("=== Benchmark Complete ===")
    report_lines.append("=" * 70)
    
    try:
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\n-> Benchmark report saved to: {report_file}")
    except Exception as e:
        print(f"\nERROR: Failed to save report to {report_file}: {e}")


def benchmark_pipeline(
    audio_file: str,
    compute_type: Optional[str] = None,
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
    output_audio: Optional[str] = None,
    report_file: Optional[str] = None,
    stt_device: Optional[str] = None,
    tts_use_gpu: bool = False,
    run_comparison: bool = False,
) -> None:
    """Run benchmark on the complete voice assistant pipeline.
    
    Args:
        audio_file: Path to input WAV audio file
        compute_type: Compute type for STT ('int8', 'float16', or 'float32')
        ollama_url: Ollama server URL
        ollama_model: Ollama model name
        output_audio: Optional path to save generated TTS audio
        report_file: Optional path to save benchmark report as text file
        stt_device: Force STT device ('cpu' or 'cuda'), auto-detects if None
        tts_use_gpu: Enable GPU for TTS (default: False)
        run_comparison: Run both CPU and GPU benchmarks for comparison
    """
    print("=" * 70)
    print("=== Voice Assistant Pipeline Benchmark ===")
    print("=" * 70)
    
    # System information
    cuda_available = check_cuda_availability()
    print(f"\nSystem: {sys.platform}")
    print(f"CUDA Available: {'Yes' if cuda_available else 'No'}")
    
    # Load audio file
    print(f"\n-> Loading audio file: {audio_file}")
    try:
        raw_bytes, sample_rate, duration = load_audio_file(audio_file)
        channels = "mono"  # We always convert to mono
        print(f"   Audio: {duration:.2f}s, {sample_rate}Hz, {channels}")
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    
    # Handle comparison mode
    if run_comparison:
        print("\n" + "=" * 70)
        print("=== COMPARISON MODE: Running CPU and GPU Benchmarks ===")
        print("=" * 70)
        
        # Run CPU benchmark
        print("\n" + ">" * 70)
        print(">>> BENCHMARK 1: CPU Mode")
        print(">" * 70)
        cpu_results = _run_single_benchmark(
            raw_bytes, sample_rate, duration, channels, audio_file,
            compute_type, ollama_url, ollama_model, output_audio, report_file,
            stt_device="cpu", tts_use_gpu=False, label="CPU"
        )
        
        # Run GPU benchmark
        print("\n" + ">" * 70)
        print(">>> BENCHMARK 2: GPU Mode")
        print(">" * 70)
        gpu_results = _run_single_benchmark(
            raw_bytes, sample_rate, duration, channels, audio_file,
            compute_type, ollama_url, ollama_model, output_audio, report_file,
            stt_device="cuda", tts_use_gpu=True, label="GPU"
        )
        
        # Print comparison summary
        _print_comparison_summary(cpu_results, gpu_results)
        
        return
    
    # Detect device for STT (if not forced)
    if stt_device is None:
        stt_device = _detect_whisper_device()
    else:
        print(f"\n-> Using forced STT device: {stt_device}")
    
    # Run single benchmark
    _run_single_benchmark(
        raw_bytes, sample_rate, duration, channels, audio_file,
        compute_type, ollama_url, ollama_model, output_audio, report_file,
        stt_device=stt_device, tts_use_gpu=tts_use_gpu, label=""
    )


def main() -> None:
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark voice assistant pipeline (STT → LLM → TTS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with auto-detection
  python benchmark.py --audio-file test.wav
  
  # Specify compute type for STT
  python benchmark.py --audio-file test.wav --compute-type int8
  
  # Use custom Ollama model and save output audio
  python benchmark.py --audio-file test.wav --ollama-model llama3.2:1b --output-audio output.wav
  
  # Save benchmark report to text file
  python benchmark.py --audio-file test.wav --report-file benchmark_report.txt
  
  # Force GPU for all components
  python benchmark.py --audio-file test.wav --stt-device cuda --tts-gpu
  
  # Compare CPU vs GPU performance
  python benchmark.py --audio-file test.wav --compare
        """,
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to input WAV audio file (required)",
    )
    
    parser.add_argument(
        "--compute-type",
        type=str,
        choices=["int8", "float16", "float32"],
        default=None,
        help="Compute type for faster-whisper. Options: int8 (Jetson/CPU), float16 (desktop GPUs), float32. "
             "If not specified, auto-detects based on device and platform.",
    )
    
    parser.add_argument(
        "--stt-device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Force STT device (cpu or cuda). If not specified, auto-detects.",
    )
    
    parser.add_argument(
        "--tts-gpu",
        action="store_true",
        help="Enable GPU acceleration for TTS (default: CPU)",
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both CPU and GPU benchmarks for comparison",
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
    
    parser.add_argument(
        "--output-audio",
        type=str,
        default=None,
        help="Optional path to save generated TTS audio (WAV format)",
    )
    
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Optional path to save benchmark report as text file",
    )
    
    args = parser.parse_args()
    
    benchmark_pipeline(
        audio_file=args.audio_file,
        compute_type=args.compute_type,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        output_audio=args.output_audio,
        report_file=args.report_file,
        stt_device=args.stt_device,
        tts_use_gpu=args.tts_gpu,
        run_comparison=args.compare,
    )


if __name__ == "__main__":
    main()

