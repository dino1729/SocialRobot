"""Integration tests using pre-recorded/generated audio files.

These tests create audio files once and reuse them for deterministic testing.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.integration import INTEGRATION_TESTS_ENABLED, SKIP_REASON
from tests.integration.base import AudioIntegrationTestBase


# Directory for cached test audio files
TEST_AUDIO_DIR = Path(__file__).parent / "test_audio_cache"


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestPrerecordedAudioSetup(AudioIntegrationTestBase):
    """Generate test audio files for use in other tests."""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        # Create cache directory
        TEST_AUDIO_DIR.mkdir(exist_ok=True)
        
        try:
            from audio.tts import KokoroTTS
            cls._tts = KokoroTTS(voice="af_bella", speed=1.0, use_gpu=False)
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load TTS: {e}")
    
    def _get_audio_path(self, name: str) -> Path:
        """Get path for a test audio file."""
        return TEST_AUDIO_DIR / f"{name}.wav"
    
    def _ensure_audio_file(self, name: str, text: str) -> Path:
        """Ensure a test audio file exists, creating if needed."""
        path = self._get_audio_path(name)
        
        if not path.exists():
            print(f"-> Generating test audio: {name}")
            audio = self._tts.synthesize(text)
            self.save_audio_wav(audio, self._tts.sample_rate, str(path))
        
        return path
    
    def test_generate_hello_world(self):
        """Generate 'hello world' test audio."""
        path = self._ensure_audio_file("hello_world", "Hello world.")
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 1000)
    
    def test_generate_question(self):
        """Generate question test audio."""
        path = self._ensure_audio_file("question", "What time is it?")
        self.assertTrue(path.exists())
    
    def test_generate_command(self):
        """Generate command test audio."""
        path = self._ensure_audio_file("command", "Set a timer for five minutes.")
        self.assertTrue(path.exists())
    
    def test_generate_long_sentence(self):
        """Generate long sentence test audio."""
        path = self._ensure_audio_file(
            "long_sentence", 
            "The quick brown fox jumps over the lazy dog near the river."
        )
        self.assertTrue(path.exists())
    
    def test_generate_numbers(self):
        """Generate numbers test audio."""
        path = self._ensure_audio_file("numbers", "One two three four five.")
        self.assertTrue(path.exists())


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestSTTWithPrerecordedAudio(AudioIntegrationTestBase):
    """Test STT with pre-recorded audio files."""
    
    _stt = None
    _test_files = {}
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        # Ensure test audio directory exists
        if not TEST_AUDIO_DIR.exists():
            raise unittest.SkipTest("Run TestPrerecordedAudioSetup first to generate test audio")
        
        try:
            from audio.stt import FasterWhisperSTT
            cls._stt = FasterWhisperSTT(
                model_size_or_path="tiny.en",
                device="cpu",
                compute_type="int8",
                language="en"
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load STT: {e}")
        
        # Map test files to expected content
        cls._test_files = {
            "hello_world": "Hello world.",
            "question": "What time is it?",
            "command": "Set a timer for five minutes.",
            "long_sentence": "The quick brown fox jumps over the lazy dog near the river.",
            "numbers": "One two three four five.",
        }
    
    def _test_file(self, name: str, expected: str, min_accuracy: float = 0.6):
        """Test transcription of a pre-recorded file."""
        path = TEST_AUDIO_DIR / f"{name}.wav"
        
        if not path.exists():
            self.skipTest(f"Test audio file not found: {path}")
        
        audio_bytes, sample_rate = self.load_audio_wav(str(path))
        transcribed = self._stt.run_stt(audio_bytes, sample_rate=sample_rate)
        
        self.assertTranscriptionSimilar(expected, transcribed, min_accuracy=min_accuracy)
    
    def test_hello_world_transcription(self):
        """Test transcription of 'hello world'."""
        # Tiny model may transcribe as "Hello White" or similar
        self._test_file("hello_world", "Hello world.", min_accuracy=0.5)
    
    def test_question_transcription(self):
        """Test transcription of question."""
        self._test_file("question", "What time is it?", min_accuracy=0.5)
    
    def test_command_transcription(self):
        """Test transcription of command."""
        self._test_file("command", "Set a timer for five minutes.", min_accuracy=0.4)
    
    def test_long_sentence_transcription(self):
        """Test transcription of long sentence."""
        self._test_file(
            "long_sentence",
            "The quick brown fox jumps over the lazy dog near the river.",
            min_accuracy=0.5
        )
    
    def test_numbers_transcription(self):
        """Test transcription of numbers."""
        # Numbers may be transcribed as digits or words
        self._test_file("numbers", "One two three four five.", min_accuracy=0.6)


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestSTTWithSyntheticAudio(AudioIntegrationTestBase):
    """Test STT with programmatically generated audio signals."""
    
    _stt = None
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        try:
            from audio.stt import FasterWhisperSTT
            cls._stt = FasterWhisperSTT(
                model_size_or_path="tiny.en",
                device="cpu",
                compute_type="int8",
                language="en"
            )
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load STT: {e}")
    
    def _generate_silence(self, duration_sec: float, sample_rate: int = 16000) -> bytes:
        """Generate silence audio."""
        num_samples = int(duration_sec * sample_rate)
        return bytes(num_samples * 2)  # 16-bit = 2 bytes per sample
    
    def _generate_noise(self, duration_sec: float, amplitude: float = 0.1, sample_rate: int = 16000) -> bytes:
        """Generate white noise audio."""
        num_samples = int(duration_sec * sample_rate)
        noise = np.random.uniform(-amplitude, amplitude, num_samples)
        noise_int16 = np.clip(noise * 32767, -32768, 32767).astype(np.int16)
        return noise_int16.tobytes()
    
    def _generate_tone(self, duration_sec: float, frequency: float, amplitude: float = 0.5, sample_rate: int = 16000) -> bytes:
        """Generate a pure sine tone."""
        num_samples = int(duration_sec * sample_rate)
        t = np.arange(num_samples) / sample_rate
        tone = amplitude * np.sin(2 * np.pi * frequency * t)
        tone_int16 = np.clip(tone * 32767, -32768, 32767).astype(np.int16)
        return tone_int16.tobytes()
    
    def test_silence_produces_empty_result(self):
        """Verify silence produces empty or minimal transcription."""
        silence = self._generate_silence(1.0)
        result = self._stt.run_stt(silence, sample_rate=16000)
        
        # Should be empty or very short (no speech)
        self.assertLess(len(result.strip()), 10)
    
    def test_pure_noise_produces_minimal_result(self):
        """Verify pure noise produces minimal transcription."""
        noise = self._generate_noise(1.0, amplitude=0.05)
        result = self._stt.run_stt(noise, sample_rate=16000)
        
        # Should be empty or minimal
        self.assertLess(len(result.strip()), 20)
    
    def test_pure_tone_produces_minimal_result(self):
        """Verify pure tone (not speech) produces minimal transcription."""
        tone = self._generate_tone(1.0, frequency=440)  # A4 note
        result = self._stt.run_stt(tone, sample_rate=16000)
        
        # Pure tone is not speech, should produce minimal output
        self.assertLess(len(result.strip()), 20)
    
    def test_handles_very_short_audio(self):
        """Verify STT handles very short audio gracefully."""
        short_audio = self._generate_silence(0.1)  # 100ms
        result = self._stt.run_stt(short_audio, sample_rate=16000)
        
        # Should not crash, may return empty
        self.assertIsInstance(result, str)
    
    def test_handles_long_silence(self):
        """Verify STT handles long silence gracefully."""
        long_silence = self._generate_silence(5.0)  # 5 seconds
        result = self._stt.run_stt(long_silence, sample_rate=16000)
        
        # Should return empty or minimal
        self.assertLess(len(result.strip()), 10)


@unittest.skipUnless(INTEGRATION_TESTS_ENABLED, SKIP_REASON)
class TestVADWithSyntheticAudio(AudioIntegrationTestBase):
    """Test VAD behavior with synthetic audio signals."""
    
    def test_vad_detects_speech_like_signal(self):
        """Test that VAD can process audio without crashing."""
        try:
            import webrtcvad
        except ImportError:
            self.skipTest("webrtcvad not installed")
        
        vad = webrtcvad.Vad(2)  # Medium aggressiveness
        
        # Generate 30ms frames at 16kHz (480 samples = 960 bytes)
        frame_samples = 480
        sample_rate = 16000
        
        # Test with silence
        silence_frame = bytes(frame_samples * 2)
        result = vad.is_speech(silence_frame, sample_rate)
        self.assertIsInstance(result, bool)
        
        # Test with noise (more likely to be detected as speech)
        noise = np.random.uniform(-0.3, 0.3, frame_samples)
        noise_int16 = np.clip(noise * 32767, -32768, 32767).astype(np.int16)
        noise_frame = noise_int16.tobytes()
        
        result = vad.is_speech(noise_frame, sample_rate)
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()

