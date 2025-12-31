"""Audio utilities for suppressing ALSA/JACK/VibeVoice warnings.

This module provides utilities to suppress the harmless but noisy warnings
from ALSA, JACK, and ML libraries when using PyAudio and TTS on Linux systems.

Usage:
    from audio.suppress_warnings import suppress_stderr, init_pyaudio_silent

    # Method 1: Import pyaudio silently at module level
    pyaudio = init_pyaudio_silent()

    # Method 2: Suppress warnings around specific operations
    with suppress_stderr():
        pa = pyaudio.PyAudio()

    # Method 3: Suppress ML library warnings (transformers, vibevoice, etc.)
    suppress_ml_warnings()
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import warnings
from contextlib import contextmanager


# Suppress ALSA error messages (must be done before importing pyaudio)
try:
    _ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
    )
    
    def _py_error_handler(filename, line, function, err, fmt):
        pass
    
    _c_error_handler = _ERROR_HANDLER_FUNC(_py_error_handler)
    _asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    _asound.snd_lib_error_set_handler(_c_error_handler)
except Exception:
    pass  # Not on Linux or libasound not available


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output from C libraries.
    
    This suppresses JACK warnings and other C library messages that
    are written directly to file descriptor 2 (stderr).
    
    Example:
        with suppress_stderr():
            pa = pyaudio.PyAudio()
            stream = pa.open(...)
    """
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


def init_pyaudio_silent():
    """Import and return pyaudio module with warnings suppressed.
    
    This function imports pyaudio while suppressing the JACK warnings
    that appear on first import.
    
    Returns:
        The pyaudio module
        
    Example:
        pyaudio = init_pyaudio_silent()
        pa = pyaudio.PyAudio()
    """
    with suppress_stderr():
        import pyaudio
    return pyaudio


# Pre-initialize pyaudio import to suppress warnings on module import
_pyaudio = None


def get_pyaudio():
    """Get the pyaudio module, initializing it silently if needed.
    
    Returns:
        The pyaudio module
    """
    global _pyaudio
    if _pyaudio is None:
        _pyaudio = init_pyaudio_silent()
    return _pyaudio


# ML library loggers to suppress (transformers, vibevoice, etc.)
_ML_LOGGERS = [
    "transformers",
    "transformers.tokenization_utils_base",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.generation.configuration_utils",
    "vibevoice",
    "diffusers",
    "accelerate",
    "huggingface_hub",
]

_ml_warnings_suppressed = False


def suppress_ml_warnings():
    """Suppress warnings from ML libraries (transformers, vibevoice, etc.).

    This function suppresses:
    - Tokenizer class mismatch warnings from transformers
    - torch_dtype deprecation warnings
    - Other harmless ML library warnings

    Call this once at the start of your program before importing ML libraries.

    Example:
        from audio.suppress_warnings import suppress_ml_warnings
        suppress_ml_warnings()

        # Now import vibevoice/transformers
        from audio.tts_vibevoice import VibeVoiceTTS
    """
    global _ml_warnings_suppressed
    if _ml_warnings_suppressed:
        return

    # Set environment variables before any imports
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Suppress Python warnings
    warnings.filterwarnings("ignore", message=".*tokenizer class you load.*")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    # Set logging levels for ML libraries
    for logger_name in _ML_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

    _ml_warnings_suppressed = True


@contextmanager
def suppress_ml_output():
    """Context manager to suppress ML library output during operations.

    Combines stderr suppression with ML warning suppression for operations
    like model loading that produce verbose output.

    Example:
        with suppress_ml_output():
            model = VibeVoiceModel.from_pretrained(...)
    """
    suppress_ml_warnings()
    with suppress_stderr():
        yield


def init_vibevoice_silent():
    """Import VibeVoice modules with warnings suppressed.

    Returns:
        tuple: (VibeVoiceStreamingForConditionalGenerationInference, VibeVoiceStreamingProcessor)

    Example:
        ModelClass, ProcessorClass = init_vibevoice_silent()
        model = ModelClass.from_pretrained(...)
    """
    suppress_ml_warnings()

    with suppress_stderr():
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

    return VibeVoiceStreamingForConditionalGenerationInference, VibeVoiceStreamingProcessor


__all__ = [
    "suppress_stderr",
    "suppress_ml_warnings",
    "suppress_ml_output",
    "init_pyaudio_silent",
    "init_vibevoice_silent",
    "get_pyaudio",
]
