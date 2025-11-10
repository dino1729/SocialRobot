"""Benchmark script comparing different STT and TTS engines."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from audio.engine_config import STTEngine, TTSEngine, create_stt_engine, create_tts_engine
from llm.ollama import OllamaClient
from benchmark import load_audio_file, save_audio_file, check_cuda_availability


# Load environment variables
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")


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
  # Test all engine combinations
  python benchmark_engines.py --audio-file test.wav --test-all
  
  # Test specific STT engine
  python benchmark_engines.py --audio-file test.wav --stt-engine openai-whisper
  
  # Test specific TTS engine
  python benchmark_engines.py --audio-file test.wav --tts-engine piper
  
  # Test specific combination
  python benchmark_engines.py --audio-file test.wav --stt-engine faster-whisper --tts-engine kokoro
  
  # Disable GPU
  python benchmark_engines.py --audio-file test.wav --test-all --no-gpu
        """,
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to input WAV audio file (required)",
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
        help="Test all engine combinations",
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
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
    
    print("=" * 70)
    print("=== VOICE ASSISTANT ENGINE BENCHMARK ===")
    print("=" * 70)
    print(f"\nSystem: {sys.platform}")
    print(f"CUDA Available: {'Yes' if check_cuda_availability() else 'No'}")
    print(f"GPU Mode: {'Disabled' if args.no_gpu else 'Enabled'}")
    print(f"Audio File: {args.audio_file}")
    
    use_gpu = not args.no_gpu
    
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
                output_dir="benchmark_outputs",
            )
            all_results.append(result)
            print("\n")
            time.sleep(1)  # Brief pause between tests
    
    # Print comparison table
    if len(all_results) > 1:
        print_comparison_table(all_results)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()

