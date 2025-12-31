"""Configuration and factory for swappable STT and TTS engines."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class STTEngine(str, Enum):
    """Available STT engine options."""
    FASTER_WHISPER = "faster-whisper"
    OPENAI_WHISPER = "openai-whisper"


class TTSEngine(str, Enum):
    """Available TTS engine options."""
    KOKORO = "kokoro"
    PIPER = "piper"
    CHATTERBOX = "chatterbox"
    VIBEVOICE = "vibevoice"


def create_stt_engine(
    engine: STTEngine | str = STTEngine.FASTER_WHISPER,
    device: Optional[str] = None,
    model_size: str = "tiny",
    compute_type: Optional[str] = None,
    language: Optional[str] = "en",
) -> Any:
    """Factory function to create STT engine.
    
    Args:
        engine: STT engine to use
        device: Device to use ('cpu' or 'cuda'), auto-detects if None
        model_size: Model size (engine-specific)
        compute_type: Compute type for faster-whisper ('int8', 'float16', 'float32')
        language: Language code
        
    Returns:
        STT engine instance
    """
    engine = STTEngine(engine)
    
    if engine == STTEngine.FASTER_WHISPER:
        from audio.stt import FasterWhisperSTT
        
        if device is None:
            device = _detect_whisper_device()
        
        if compute_type is None:
            compute_type = _get_default_compute_type(device)
        
        # Map generic model sizes to faster-whisper format
        model_map = {
            "tiny": "tiny.en" if language == "en" else "tiny",
            "base": "base.en" if language == "en" else "base",
            "small": "small.en" if language == "en" else "small",
            "medium": "medium.en" if language == "en" else "medium",
            "large": "large-v3",
        }
        model_name = model_map.get(model_size, model_size)
        
        return FasterWhisperSTT(
            model_size_or_path=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
        )
    
    elif engine == STTEngine.OPENAI_WHISPER:
        from audio.stt_openai import OpenAIWhisperSTT
        
        return OpenAIWhisperSTT(
            model_size=model_size,
            device=device,
            language=language,
        )
    
    else:
        raise ValueError(f"Unknown STT engine: {engine}")


def create_tts_engine(
    engine: TTSEngine | str = TTSEngine.KOKORO,
    use_gpu: bool = False,
    voice: Optional[str] = None,
    speed: float = 1.0,
    voice_path: Optional[str] = None,
    speaker: Optional[str] = None,
    cfg_scale: float = 1.5,
    use_turbo: bool = True,
    **kwargs: Any,
) -> Any:
    """Factory function to create TTS engine.

    Args:
        engine: TTS engine to use
        use_gpu: Enable GPU acceleration
        voice: Voice to use (engine-specific, for Kokoro)
        speed: Speech speed multiplier (for Kokoro)
        voice_path: Path to voice WAV file for zero-shot cloning (for Chatterbox)
        speaker: Speaker name for VibeVoice (default: Carter)
        cfg_scale: Classifier-free guidance scale for VibeVoice (default: 1.5)
        use_turbo: Use Chatterbox turbo model for faster inference (default: True)
        **kwargs: Additional engine-specific arguments

    Returns:
        TTS engine instance
    """
    engine = TTSEngine(engine)

    if engine == TTSEngine.KOKORO:
        from audio.tts import KokoroTTS

        return KokoroTTS(
            voice=voice or "af_bella",
            speed=speed,
            use_gpu=use_gpu,
            **kwargs,
        )

    elif engine == TTSEngine.PIPER:
        from audio.tts_piper import PiperTTS

        return PiperTTS(
            use_gpu=use_gpu,
            **kwargs,
        )

    elif engine == TTSEngine.CHATTERBOX:
        from audio.tts_chatterbox import ChatterboxTTS

        return ChatterboxTTS(
            voice_path=voice_path,
            use_gpu=use_gpu,
            use_turbo=use_turbo,
            **kwargs,
        )

    elif engine == TTSEngine.VIBEVOICE:
        from audio.tts_vibevoice import VibeVoiceTTS

        return VibeVoiceTTS(
            speaker=speaker or "Carter",
            use_gpu=use_gpu,
            cfg_scale=cfg_scale,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown TTS engine: {engine}")


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
    import os
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


# =============================================================================
# WER (Word Error Rate) utilities for benchmarking
# =============================================================================

def normalize_text_for_wer(text: str) -> str:
    """Normalize text for WER computation.
    
    Applies the following normalizations:
    - Lowercase
    - Remove punctuation
    - Collapse multiple spaces to single space
    - Strip leading/trailing whitespace
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text suitable for WER comparison
    """
    import re
    
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation (keep only alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", "", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    # Strip
    return text.strip()


def _levenshtein_distance(ref_words: list, hyp_words: list) -> int:
    """Compute Levenshtein (edit) distance between two word sequences.
    
    Uses dynamic programming to find minimum number of insertions,
    deletions, and substitutions needed to transform hyp_words into ref_words.
    
    Args:
        ref_words: Reference word list
        hyp_words: Hypothesis word list
        
    Returns:
        Minimum edit distance (number of operations)
    """
    m, n = len(ref_words), len(hyp_words)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: transforming empty string
    for i in range(m + 1):
        dp[i][0] = i  # deletions
    for j in range(n + 1):
        dp[0][j] = j  # insertions
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # no operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )
    
    return dp[m][n]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis.
    
    WER = (S + D + I) / N
    where:
      S = substitutions
      D = deletions
      I = insertions
      N = number of words in reference
    
    Both strings are normalized before comparison.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted/transcribed text
        
    Returns:
        WER as a float (0.0 = perfect match, 1.0 = all words wrong).
        Can exceed 1.0 if hypothesis has many insertions.
    """
    # Normalize both texts
    ref_norm = normalize_text_for_wer(reference)
    hyp_norm = normalize_text_for_wer(hypothesis)
    
    # Handle edge cases
    if not ref_norm and not hyp_norm:
        return 0.0  # both empty is a perfect match
    if not ref_norm:
        # Reference is empty but hypothesis has words - infinite WER conceptually
        # Return length of hypothesis as WER (all insertions)
        return float(len(hyp_norm.split()))
    if not hyp_norm:
        # Hypothesis is empty - 100% deletion rate
        return 1.0
    
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    
    distance = _levenshtein_distance(ref_words, hyp_words)
    
    return distance / len(ref_words)


__all__ = [
    "STTEngine",
    "TTSEngine",
    "create_stt_engine",
    "create_tts_engine",
    "normalize_text_for_wer",
    "word_error_rate",
]

