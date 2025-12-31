"""Integration tests for Kokoro TTS + OpenAI Whisper STT pipeline."""

import os
import tempfile
import unittest

import numpy as np

from tests.integration import INTEGRATION_TESTS_ENABLED, SKIP_REASON
from tests.integration.base import AudioIntegrationTestBase


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestKokoroOpenAIWhisperRoundtrip(AudioIntegrationTestBase):
    """Round-trip tests: Kokoro TTS generates audio, OpenAI Whisper transcribes it."""
    
    _tts = None
    _stt = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        try:
            from audio.tts import KokoroTTS
            from audio.stt_openai import OpenAIWhisperSTT
            
            print("-> Loading Kokoro TTS model...")
            cls._tts = KokoroTTS(voice="af_bella", speed=1.0, use_gpu=False)
            
            print("-> Loading OpenAI Whisper STT model...")
            cls._stt = OpenAIWhisperSTT(
                model_size="tiny",
                device="cpu",
                language="en"
            )
        except ImportError as e:
            raise unittest.SkipTest(f"Missing dependency: {e}. Install with: pip install openai-whisper")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load models: {e}")
    
    def test_simple_phrase_roundtrip(self):
        """Test TTS->STT roundtrip with a simple phrase."""
        original_text = self.QUICK_PHRASE
        
        audio_data = self._tts.synthesize(original_text)
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.8)
    
    def test_question_phrase_roundtrip(self):
        """Test roundtrip with a question."""
        original_text = "What time is it right now?"
        
        audio_data = self._tts.synthesize(original_text)
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.7)
    
    def test_command_phrase_roundtrip(self):
        """Test roundtrip with a command."""
        original_text = "Play some music please."
        
        audio_data = self._tts.synthesize(original_text)
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.7)
    
    def test_multiple_phrases(self):
        """Test roundtrip with multiple phrases."""
        for phrase in self.TEST_PHRASES:
            with self.subTest(phrase=phrase):
                audio_data = self._tts.synthesize(phrase)
                audio_bytes = self.audio_to_bytes(audio_data)
                transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
                
                self.assertTranscriptionSimilar(phrase, transcribed, min_accuracy=0.5)


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestOpenAIWhisperSTT(AudioIntegrationTestBase):
    """Tests for OpenAI Whisper STT."""
    
    _stt = None
    _tts = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        try:
            from audio.stt_openai import OpenAIWhisperSTT
            from audio.tts import KokoroTTS
            
            cls._stt = OpenAIWhisperSTT(model_size="tiny", device="cpu", language="en")
            cls._tts = KokoroTTS(voice="af_bella", speed=1.0, use_gpu=False)
        except ImportError as e:
            raise unittest.SkipTest(f"Missing dependency: {e}")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load models: {e}")
    
    def test_empty_audio_returns_empty(self):
        """Verify empty audio returns empty string."""
        result = self._stt.run_stt(b'', sample_rate=16000)
        self.assertEqual(result, "")
    
    def test_transcribes_clear_speech(self):
        """Verify clear speech is transcribed correctly."""
        audio = self._tts.synthesize("This is a test.")
        audio_bytes = self.audio_to_bytes(audio)
        
        result = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertIn("test", result.lower())


if __name__ == "__main__":
    unittest.main()

