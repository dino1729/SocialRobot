"""Integration tests for Piper TTS + STT pipelines."""

import os
import tempfile
import unittest

import numpy as np

from tests.integration import INTEGRATION_TESTS_ENABLED, SKIP_REASON
from tests.integration.base import AudioIntegrationTestBase


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestPiperFasterWhisperRoundtrip(AudioIntegrationTestBase):
    """Round-trip tests: Piper TTS generates audio, faster-whisper transcribes it."""
    
    _tts = None
    _stt = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        try:
            from audio.tts_piper import PiperTTS
            from audio.stt import FasterWhisperSTT
            
            print("-> Loading Piper TTS model...")
            cls._tts = PiperTTS(use_gpu=False)
            
            print("-> Loading faster-whisper STT model...")
            cls._stt = FasterWhisperSTT(
                model_size_or_path="tiny.en",
                device="cpu",
                compute_type="int8",
                language="en"
            )
        except ImportError as e:
            raise unittest.SkipTest(f"Missing dependency: {e}. Install with: pip install piper-tts")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load models: {e}")
    
    def test_simple_phrase_roundtrip(self):
        """Test TTS->STT roundtrip with a simple phrase."""
        original_text = self.QUICK_PHRASE
        
        audio_data = self._tts.synthesize(original_text)
        self.assertGreater(len(audio_data), 0, "TTS should produce audio")
        
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        # Piper with tiny STT may have lower accuracy
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.4)
    
    def test_question_roundtrip(self):
        """Test roundtrip with a question."""
        original_text = "What is your name?"
        
        audio_data = self._tts.synthesize(original_text)
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.6)
    
    def test_multiple_phrases(self):
        """Test roundtrip with multiple phrases."""
        for phrase in self.TEST_PHRASES[:2]:  # Test fewer for speed
            with self.subTest(phrase=phrase):
                audio_data = self._tts.synthesize(phrase)
                audio_bytes = self.audio_to_bytes(audio_data)
                transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
                
                self.assertTranscriptionSimilar(phrase, transcribed, min_accuracy=0.5)


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestPiperTTSOutput(AudioIntegrationTestBase):
    """Tests for Piper TTS output quality."""
    
    _tts = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        try:
            from audio.tts_piper import PiperTTS
            cls._tts = PiperTTS(use_gpu=False)
        except ImportError as e:
            raise unittest.SkipTest(f"Missing dependency: {e}")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load TTS: {e}")
    
    def test_generates_audio(self):
        """Verify TTS produces non-empty audio output."""
        audio = self._tts.synthesize("Hello world.")
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_audio_in_valid_range(self):
        """Verify audio samples are in valid range."""
        audio = self._tts.synthesize("Testing audio range.")
        
        self.assertLessEqual(np.max(np.abs(audio)), 1.5)  # Allow some headroom
    
    def test_audio_has_content(self):
        """Verify audio has actual content (not silence)."""
        audio = self._tts.synthesize("This should not be silence.")
        
        rms = np.sqrt(np.mean(audio ** 2))
        self.assertGreater(rms, 0.001, "Audio should have audible content")
    
    def test_empty_text_returns_empty(self):
        """Verify empty text returns empty audio."""
        audio = self._tts.synthesize("")
        self.assertEqual(len(audio), 0)
    
    def test_sample_rate_is_valid(self):
        """Verify sample rate is standard."""
        # Piper typically uses 22050 Hz
        self.assertIn(self._tts.sample_rate, [16000, 22050, 24000, 44100, 48000])


if __name__ == "__main__":
    unittest.main()

