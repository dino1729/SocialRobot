"""Integration tests for Kokoro TTS + faster-whisper STT pipeline."""

import os
import tempfile
import unittest

import numpy as np

from tests.integration import INTEGRATION_TESTS_ENABLED, SKIP_REASON
from tests.integration.base import AudioIntegrationTestBase, skip_if_missing_dependency


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestKokoroFasterWhisperRoundtrip(AudioIntegrationTestBase):
    """Round-trip tests: Kokoro TTS generates audio, faster-whisper transcribes it."""
    
    _tts = None
    _stt = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        # Import and initialize engines (done once per test class)
        try:
            from audio.tts import KokoroTTS
            from audio.stt import FasterWhisperSTT
            
            print("-> Loading Kokoro TTS model...")
            cls._tts = KokoroTTS(voice="af_bella", speed=1.0, use_gpu=False)
            
            print("-> Loading faster-whisper STT model...")
            cls._stt = FasterWhisperSTT(
                model_size_or_path="tiny.en",
                device="cpu",
                compute_type="int8",
                language="en"
            )
        except ImportError as e:
            raise unittest.SkipTest(f"Missing dependency: {e}")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load models: {e}")
    
    def test_simple_phrase_roundtrip(self):
        """Test TTS->STT roundtrip with a simple phrase."""
        original_text = self.QUICK_PHRASE
        
        # Generate audio
        audio_data = self._tts.synthesize(original_text)
        self.assertGreater(len(audio_data), 0, "TTS should produce audio")
        
        # Convert to bytes for STT
        audio_bytes = self.audio_to_bytes(audio_data)
        
        # Transcribe
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        # Verify (tiny model may have lower accuracy)
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.5)
    
    def test_question_phrase_roundtrip(self):
        """Test roundtrip with a question."""
        original_text = "What is the weather like today?"
        
        audio_data = self._tts.synthesize(original_text)
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.7)
    
    def test_command_phrase_roundtrip(self):
        """Test roundtrip with a command-style phrase."""
        original_text = "Set a timer for five minutes."
        
        audio_data = self._tts.synthesize(original_text)
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.7)
    
    def test_longer_sentence_roundtrip(self):
        """Test roundtrip with a longer sentence."""
        original_text = "The quick brown fox jumps over the lazy dog near the riverbank."
        
        audio_data = self._tts.synthesize(original_text)
        audio_bytes = self.audio_to_bytes(audio_data)
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertTranscriptionSimilar(original_text, transcribed, min_accuracy=0.6)
    
    def test_multiple_phrases(self):
        """Test roundtrip with multiple different phrases."""
        for phrase in self.TEST_PHRASES:
            with self.subTest(phrase=phrase):
                audio_data = self._tts.synthesize(phrase)
                audio_bytes = self.audio_to_bytes(audio_data)
                transcribed = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
                
                # Use lower threshold since some phrases are harder
                self.assertTranscriptionSimilar(phrase, transcribed, min_accuracy=0.5)


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestKokoroTTSOutput(AudioIntegrationTestBase):
    """Tests for Kokoro TTS output quality."""
    
    _tts = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        try:
            from audio.tts import KokoroTTS
            cls._tts = KokoroTTS(voice="af_bella", speed=1.0, use_gpu=False)
        except ImportError as e:
            raise unittest.SkipTest(f"Missing dependency: {e}")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load TTS: {e}")
    
    def test_generates_audio(self):
        """Verify TTS produces non-empty audio output."""
        audio = self._tts.synthesize("Hello world.")
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
        self.assertEqual(audio.dtype, np.float32)
    
    def test_audio_in_valid_range(self):
        """Verify audio samples are in valid range [-1, 1]."""
        audio = self._tts.synthesize("Testing audio range.")
        
        self.assertLessEqual(np.max(audio), 1.0)
        self.assertGreaterEqual(np.min(audio), -1.0)
    
    def test_audio_has_content(self):
        """Verify audio has actual content (not silence)."""
        audio = self._tts.synthesize("This should not be silence.")
        
        # RMS should be above noise floor
        rms = np.sqrt(np.mean(audio ** 2))
        self.assertGreater(rms, 0.01, "Audio should have audible content")
    
    def test_empty_text_returns_empty(self):
        """Verify empty text returns empty audio."""
        audio = self._tts.synthesize("")
        self.assertEqual(len(audio), 0)
    
    def test_whitespace_text_returns_empty(self):
        """Verify whitespace-only text returns empty audio."""
        audio = self._tts.synthesize("   ")
        self.assertEqual(len(audio), 0)
    
    def test_longer_text_produces_longer_audio(self):
        """Verify longer text produces proportionally longer audio."""
        short_audio = self._tts.synthesize("Hi.")
        long_audio = self._tts.synthesize("Hello, how are you doing today? I hope you are well.")
        
        # Long audio should be at least 2x longer
        self.assertGreater(len(long_audio), len(short_audio) * 2)
    
    def test_save_and_load_wav(self):
        """Test saving audio to WAV and loading it back."""
        original_audio = self._tts.synthesize("Test saving to file.")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            self.save_audio_wav(original_audio, self._tts.sample_rate, filepath)
            
            # Verify file exists and has content
            self.assertTrue(os.path.exists(filepath))
            self.assertGreater(os.path.getsize(filepath), 0)
            
            # Load back
            audio_bytes, sample_rate = self.load_audio_wav(filepath)
            
            self.assertEqual(sample_rate, self._tts.sample_rate)
            self.assertGreater(len(audio_bytes), 0)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestFasterWhisperSTTInput(AudioIntegrationTestBase):
    """Tests for faster-whisper STT input handling."""
    
    _stt = None
    _tts = None  # Used to generate test audio
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        try:
            from audio.stt import FasterWhisperSTT
            from audio.tts import KokoroTTS
            
            cls._stt = FasterWhisperSTT(
                model_size_or_path="tiny.en",
                device="cpu",
                compute_type="int8",
                language="en"
            )
            cls._tts = KokoroTTS(voice="af_bella", speed=1.0, use_gpu=False)
        except ImportError as e:
            raise unittest.SkipTest(f"Missing dependency: {e}")
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load models: {e}")
    
    def test_empty_audio_returns_empty(self):
        """Verify empty audio returns empty string."""
        result = self._stt.run_stt(b'', sample_rate=16000)
        self.assertEqual(result, "")
    
    def test_silence_returns_empty_or_minimal(self):
        """Verify silence produces empty or minimal transcription."""
        # 1 second of silence
        silence = bytes(32000)  # 16kHz * 2 bytes
        
        result = self._stt.run_stt(silence, sample_rate=16000)
        
        # Should be empty or very short
        self.assertLess(len(result), 10)
    
    def test_handles_different_sample_rates(self):
        """Verify STT handles audio at TTS sample rate."""
        audio = self._tts.synthesize("Hello there.")
        audio_bytes = self.audio_to_bytes(audio)
        
        # TTS might output at different sample rate than 16kHz
        result = self._stt.run_stt(audio_bytes, sample_rate=self._tts.sample_rate)
        
        self.assertIn("hello", result.lower())
    
    def test_transcribes_from_wav_file(self):
        """Test transcription from a saved WAV file."""
        audio = self._tts.synthesize("Testing file input.")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            filepath = f.name
        
        try:
            self.save_audio_wav(audio, self._tts.sample_rate, filepath)
            audio_bytes, sample_rate = self.load_audio_wav(filepath)
            
            result = self._stt.run_stt(audio_bytes, sample_rate=sample_rate)
            
            self.assertIn("test", result.lower())
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    unittest.main()

