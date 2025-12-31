"""Base classes and utilities for integration tests."""

import os
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from tests.integration import INTEGRATION_TESTS_ENABLED, SKIP_REASON


class AudioIntegrationTestBase(unittest.TestCase):
    """Base class for audio integration tests with common utilities."""
    
    # Test phrases for round-trip testing (varied complexity)
    TEST_PHRASES = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "What is the weather like in San Francisco?",
        "Please set a timer for five minutes.",
    ]
    
    # Short phrase for quick tests
    QUICK_PHRASE = "Hello world."
    
    @classmethod
    def setUpClass(cls):
        """Skip all tests if integration tests are disabled."""
        if not INTEGRATION_TESTS_ENABLED:
            raise unittest.SkipTest(SKIP_REASON)
    
    def save_audio_wav(self, audio_data: np.ndarray, sample_rate: int, filepath: str) -> None:
        """Save audio data as WAV file.
        
        Args:
            audio_data: Float32 audio data (-1.0 to 1.0)
            sample_rate: Sample rate in Hz
            filepath: Output WAV file path
        """
        # Convert to int16
        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def load_audio_wav(self, filepath: str) -> tuple[bytes, int]:
        """Load WAV file as raw bytes.
        
        Args:
            filepath: Path to WAV file
            
        Returns:
            Tuple of (raw_bytes, sample_rate)
        """
        with wave.open(filepath, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            raw_bytes = wav_file.readframes(wav_file.getnframes())
        return raw_bytes, sample_rate
    
    def audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert float32 audio to int16 bytes.
        
        Args:
            audio_data: Float32 audio data (-1.0 to 1.0)
            
        Returns:
            Raw audio bytes in int16 format
        """
        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    # Number word to digit mapping for normalization
    NUMBER_WORDS = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12',
    }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (lowercase, strip punctuation, normalize numbers).
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        import re
        # Remove punctuation and extra whitespace, lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = ' '.join(text.split())
        
        # Normalize number words to digits for consistent comparison
        words = text.split()
        normalized_words = []
        for word in words:
            if word in self.NUMBER_WORDS:
                normalized_words.append(self.NUMBER_WORDS[word])
            elif word.isdigit():
                normalized_words.append(word)
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def calculate_word_accuracy(self, expected: str, actual: str) -> float:
        """Calculate word-level accuracy between expected and actual text.
        
        Args:
            expected: Expected text
            actual: Actual transcribed text
            
        Returns:
            Accuracy as float (0.0 to 1.0)
        """
        expected_words = set(self.normalize_text(expected).split())
        actual_words = set(self.normalize_text(actual).split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
        
        matches = len(expected_words & actual_words)
        return matches / len(expected_words)
    
    def assertTranscriptionSimilar(
        self, 
        expected: str, 
        actual: str, 
        min_accuracy: float = 0.7,
        msg: Optional[str] = None
    ) -> None:
        """Assert that transcription is similar enough to expected text.
        
        Args:
            expected: Expected text
            actual: Actual transcribed text
            min_accuracy: Minimum word accuracy required (default 70%)
            msg: Optional assertion message
        """
        accuracy = self.calculate_word_accuracy(expected, actual)
        
        if msg is None:
            msg = (
                f"Transcription accuracy {accuracy:.1%} below threshold {min_accuracy:.1%}\n"
                f"Expected: '{expected}'\n"
                f"Actual: '{actual}'"
            )
        
        self.assertGreaterEqual(accuracy, min_accuracy, msg)


def skip_if_missing_dependency(module_name: str, package_hint: str = None):
    """Decorator to skip test if a dependency is missing.
    
    Args:
        module_name: Name of the module to check
        package_hint: Install hint (defaults to module_name)
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            try:
                __import__(module_name)
            except ImportError:
                raise unittest.SkipTest(
                    f"Missing dependency: {module_name}. "
                    f"Install with: pip install {package_hint or module_name}"
                )
            return test_func(*args, **kwargs)
        wrapper.__name__ = test_func.__name__
        wrapper.__doc__ = test_func.__doc__
        return wrapper
    return decorator

