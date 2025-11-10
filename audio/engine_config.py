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
    **kwargs: Any,
) -> Any:
    """Factory function to create TTS engine.
    
    Args:
        engine: TTS engine to use
        use_gpu: Enable GPU acceleration
        voice: Voice to use (engine-specific)
        speed: Speech speed multiplier
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


__all__ = [
    "STTEngine",
    "TTSEngine",
    "create_stt_engine",
    "create_tts_engine",
]

