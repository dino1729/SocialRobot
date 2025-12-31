"""Integration tests for Chatterbox TTS + STT pipelines."""

import os
import tempfile
import unittest

import numpy as np

from tests.integration import INTEGRATION_TESTS_ENABLED, SKIP_REASON
from tests.integration.base import AudioIntegrationTestBase


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestChatterboxFasterWhisperRoundtrip(AudioIntegrationTestBase):
    """Round-trip tests: Chatterbox TTS generates audio, faster-whisper transcribes it."""
    
    _tts = None
    _stt = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        try:
            from audio.tts_chatterbox import ChatterboxTTS
            from audio.stt import FasterWhisperSTT
            
            print("-> Loading Chatterbox TTS model (this may take a while)...")
            # Use default voice (no voice_path), turbo for speed
            cls._tts = ChatterboxTTS(
                voice_path=None,
                use_gpu=False,
                use_turbo=True
            )
            
            print("-> Loading faster-whisper STT model...")
            cls._stt = FasterWhisperSTT(
                model_size_or_path="tiny.en",
                device="cpu",
                compute_type="int8",
                language="en"
            )
        except ImportError as e:
            raise unittest.SkipTest(f"Missing dependency: {e}. Install with: pip install chatterbox-tts")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load models: {e}")
    
    def test_simple_phrase_roundtrip(self):
        """Test TTS->STT roundtrip with a simple phrase."""
        original_text = self.QUICK_PHRASE
        
        audio_data = self._tts.synthesize(original_text)
        self.assertGreater(len(audio_data), 0, "TTS should produce audio")
        
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        # Chatterbox with tiny STT may have lower accuracy
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.4)
    
    def test_question_roundtrip(self):
        """Test roundtrip with a question."""
        original_text = "How are you doing today?"
        
        audio_data = self._tts.synthesize(original_text)
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.5)


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestChatterboxTTSOutput(AudioIntegrationTestBase):
    """Tests for Chatterbox TTS output quality."""
    
    _tts = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        try:
            from audio.tts_chatterbox import ChatterboxTTS
            cls._tts = ChatterboxTTS(voice_path=None, use_gpu=False, use_turbo=True)
        except ImportError as e:
            raise unittest.SkipTest(f"Missing dependency: {e}")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load TTS: {e}")
    
    def test_generates_audio(self):
        """Verify TTS produces non-empty audio output."""
        audio = self._tts.synthesize("Hello world.")
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
    
    def test_audio_has_content(self):
        """Verify audio has actual content (not silence)."""
        audio = self._tts.synthesize("This should produce speech.")
        
        rms = np.sqrt(np.mean(audio ** 2))
        self.assertGreater(rms, 0.001, "Audio should have audible content")
    
    def test_empty_text_returns_empty(self):
        """Verify empty text returns empty audio."""
        audio = self._tts.synthesize("")
        self.assertEqual(len(audio), 0)
    
    def test_sample_rate_is_valid(self):
        """Verify sample rate is standard."""
        # Chatterbox uses 24000 Hz
        self.assertEqual(self._tts.sample_rate, 24000)
    
    def test_handles_longer_text(self):
        """Verify TTS handles longer text (sentence chunking)."""
        long_text = (
            "This is a longer piece of text that should be handled properly. "
            "It contains multiple sentences to test the chunking feature."
        )
        
        audio = self._tts.synthesize(long_text)
        self.assertGreater(len(audio), 0)
        
        # Longer text should produce longer audio
        short_audio = self._tts.synthesize("Hi.")
        self.assertGreater(len(audio), len(short_audio) * 2)


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestChatterboxWithVoiceCloning(AudioIntegrationTestBase):
    """Tests for Chatterbox TTS with voice cloning (if voice file available)."""
    
    _tts = None
    _stt = None
    _voice_path = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        # Check if any voice file exists
        voices_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'voices')
        if os.path.exists(voices_dir):
            voice_files = [f for f in os.listdir(voices_dir) if f.endswith('.wav')]
            if voice_files:
                cls._voice_path = os.path.join(voices_dir, voice_files[0])
        
        if not cls._voice_path or not os.path.exists(cls._voice_path):
            raise unittest.SkipTest("No voice file available for voice cloning test")
        
        try:
            from audio.tts_chatterbox import ChatterboxTTS
            from audio.stt import FasterWhisperSTT
            
            print(f"-> Loading Chatterbox with voice: {cls._voice_path}")
            cls._tts = ChatterboxTTS(
                voice_path=cls._voice_path,
                use_gpu=False,
                use_turbo=True
            )
            cls._stt = FasterWhisperSTT(
                model_size_or_path="tiny.en",
                device="cpu",
                compute_type="int8"
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load models: {e}")
    
    def test_voice_cloning_produces_audio(self):
        """Verify voice cloning produces audio."""
        audio = self._tts.synthesize("Testing voice cloning.")
        
        self.assertGreater(len(audio), 0)
        self.assertIsInstance(audio, np.ndarray)
    
    def test_voice_cloning_roundtrip(self):
        """Test roundtrip with cloned voice."""
        original_text = "Hello, this is a cloned voice test."
        
        audio = self._tts.synthesize(original_text)
        audio_bytes = self.audio_to_bytes(audio)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        # Voice cloning may have lower accuracy
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.4)


if __name__ == "__main__":
    unittest.main()

